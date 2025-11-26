import sys
import os
import threading
import time
import socket
import pickle
import hashlib
import shutil

import numpy as np
# imghdr 已弃用：优先使用 Pillow 识别图片格式，无法导入时回退到简单魔数字检测
try:
    from PIL import Image
    import io

    def _img_type_bytes(h: bytes):
        try:
            bio = io.BytesIO(h[:512])
            im = Image.open(bio)
            fmt = (im.format or '').lower()
            if fmt == 'jpg':
                fmt = 'jpeg'
            return fmt or None
        except Exception:
            return None
except Exception:
    Image = None
    def _img_type_bytes(h: bytes):
        try:
            b = h if isinstance(h, (bytes, bytearray)) else bytes(h or b'')
            if b.startswith(b'\x89PNG\r\n\x1a\n'):
                return 'png'
            if b.startswith(b'\xff\xd8'):
                return 'jpeg'
            if b[:6] in (b'GIF87a', b'GIF89a'):
                return 'gif'
            if b.startswith(b'BM'):
                return 'bmp'
            if len(b) >= 12 and b[8:12] == b'WEBP':
                return 'webp'
        except Exception:
            pass
        return None

from typing import Optional
from flask import request, jsonify, send_from_directory

# 解决 backend 模块导入问题（稳健地添加项目根目录到 sys.path）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # frontend 文件夹
PROJECT_ROOT = os.path.dirname(BASE_DIR)                         # 项目根目录
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from flask import Flask, render_template
from backend.tcp_communication import TCPSender, TCPReceiver, TCPMessage, MessageType
from backend.file_handler import FileProcessor
from backend.channel_coding import (
    ChannelEncoder,
    bits_to_bytes,
    bytes_to_bits,
)
from backend.channel_simulator import BSCChannel
from backend.crc_checker import ErrorDetectionSystem, CRCCalculator

# 将 Flask 的模板/静态目录设置为 frontend 目录，使用绝对路径的上传/下载目录
# 正确映射静态目录到 frontend/static，避免 /static 路由指向错误目录导致资源 404
app = Flask(__name__, template_folder=BASE_DIR, static_folder=os.path.join(BASE_DIR, 'static'))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
# 使用用户主目录的 Downloads 作为默认接收目录（与多数浏览器默认下载目录一致）
app.config['DOWNLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}

# 确保上传和下载文件夹存在（使用绝对路径）
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

# 全局状态管理
app_state = {
    'sender_connected': False,
    'receiver_connected': False,
    'sending_in_progress': False,
    'receiving_in_progress': False,
    'progress': 0,
    'status': '就绪',
    'status_level': 'info',
    'stats': {
        'bytes_sent': 0,
        'messages_sent': 0,
        'retrans_count': 0,
        'bytes_received': 0,
        'messages_received': 0,
        'errors_detected': 0
    },
    'encoding_method': None,
    'error_rate': 0.0,
    'current_receive': {
        'buffer': bytearray(),
        'expected_crc': None,
        'file_name': None,
        'encoding_method': None,
        'start_ts': None,
    },
    'received_files': [],
    # 最近接收的文件 hash（用于去重），按时间窗口限制长度
    'received_hashes': [],
    # 未完成分片消息的临时信息：message_id -> {parts{}, total_parts, last_update, filename, sha256_total}
    'incomplete_messages': {},
    # 去抖：最后一条状态及时间
    'last_status': '',
    'last_status_ts': 0,
    # 兼容后续逻辑所需字段
    'last_status_time': 0,
    # ephemeral 缓冲区（用于接收无头部的连续碎片，静默超时后合并一次性落盘）
    'ephemeral_buffers': {'current': None},
    # 性能监控历史
    'performance_history': [],
    # 发送/接收速率采样的上次快照
    'perf_last': {
        'ts': 0.0,
        'bytes_sent': 0,
        'bytes_received': 0
    },
    # 记录本次发送的原始文件名
    'sending_filename': None,
    # 记录最近上传的文件名（web 端）
    'last_uploaded_name': None,
    # 状态日志历史（用于前端显示）
    'status_logs': [],
    # 记录最近一次发送的完整文件路径（用于重传）
    'sending_filepath': None,
    # 待用户确认的重传提示
    'pending_retrans': None,
}

# 线程锁和实例变量
state_lock = threading.Lock()
tcp_sender = None
tcp_receiver = None
channel_encoder = ChannelEncoder()
file_processor = FileProcessor()
error_detection = ErrorDetectionSystem()
crc_calculator = CRCCalculator("CRC-32")
sender_worker = None


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    # 将后端默认下载目录注入模板，前端显示为默认保存目录
    return render_template('index.html', downloads_dir=app.config['DOWNLOAD_FOLDER'])


@app.route('/api/status')
def get_status():
    with state_lock:
        # 性能历史增量采样：KB/s
        try:
            now = time.time()
            last = app_state.setdefault('perf_last', {'ts': 0.0, 'bytes_sent': 0, 'bytes_received': 0})
            dt = max(1e-6, now - (last.get('ts') or 0.0))
            cur_sent = app_state['stats'].get('bytes_sent', 0)
            cur_recv = app_state['stats'].get('bytes_received', 0)
            d_sent = max(0, cur_sent - (last.get('bytes_sent') or 0))
            d_recv = max(0, cur_recv - (last.get('bytes_received') or 0))
            speed_kbs = round((d_sent + d_recv) / dt / 1024.0, 2)

            hist = app_state.setdefault('performance_history', [])
            hist.append({
                'time': time.strftime('%H:%M:%S'),
                'speed': speed_kbs,   # KB/s
                'error_rate': 0.0     # 暂无误码统计时置0
            })
            if len(hist) > 50:
                hist.pop(0)

            last['ts'] = now
            last['bytes_sent'] = cur_sent
            last['bytes_received'] = cur_recv
        except Exception:
            pass

        return jsonify({
            'progress': app_state.get('progress', 0),
            'status': app_state.get('status', '就绪'),
            'status_level': app_state.get('status_level', 'info'),
            'sender_connected': app_state.get('sender_connected', False),
            'receiver_connected': app_state.get('receiver_connected', False),
            'sending_in_progress': app_state.get('sending_in_progress', False),
            'receiving_in_progress': app_state.get('receiving_in_progress', False),
            'stats': app_state.get('stats', {}),
            'encoding_method': app_state.get('encoding_method', None),
            'error_rate': app_state.get('error_rate', 0.0),
            'received_files': app_state.get('received_files', []),
            'listening_port': app_state.get('listening_port', None),
            'performance_history': app_state.get('performance_history', []),
            'status_logs': app_state.get('status_logs', [])[-20:],  # 只返回最近20条日志
            'pending_retrans': app_state.get('pending_retrans')
        })


@app.route('/api/connect/sender', methods=['POST'])
def connect_sender():
    global tcp_sender
    data = request.json
    ip = data.get('ip')
    try:
        port = int(data.get('port') or 0)
    except Exception:
        port = 0
    encoding_method = data.get('encoding_method')
    # 误码率支持0.001精度（千分之一）
    error_rate = data.get('error_rate', app_state.get('error_rate', 0.0))

    try:
        # 创建 TCPSender
        tcp_sender = TCPSender()
        tcp_sender.encoding_method = encoding_method

        # 发送端回调接线：实时更新发送统计/状态
        def _on_data_sent(nbytes: int):
            try:
                with state_lock:
                    app_state['stats']['bytes_sent'] = app_state['stats'].get('bytes_sent', 0) + int(nbytes or 0)
                    app_state['stats']['messages_sent'] = app_state['stats'].get('messages_sent', 0) + 1
            except Exception:
                pass

        def _on_status_update(msg: str):
            _set_status(msg, 'info')

        def _on_retx():
            try:
                with state_lock:
                    fp = app_state.get('sending_filepath')
                    enc = app_state.get('encoding_method') or encoding_method
                    ber = app_state.get('error_rate', error_rate)
                if not fp or not os.path.exists(fp):
                    _set_status('收到重传请求，但找不到上次发送的文件', 'warning')
                    return
                global sender_worker
                if sender_worker and sender_worker.is_alive():
                    _set_status('收到重传请求，但当前仍在发送中，忽略', 'warning')
                    return
                _set_status('收到重传请求，开始自动重传...', 'warning')
                with state_lock:
                    app_state['stats']['retrans_count'] = app_state['stats'].get('retrans_count', 0) + 1
                sender_worker = WebFileSenderWorker(
                    sender=tcp_sender,
                    file_path=fp,
                    encoding_method=enc,
                    error_rate=ber,
                    chunk_size=4200
                )
                sender_worker.start()
            except Exception as e:
                _set_status(f'自动重传启动失败: {e}', 'danger')

        tcp_sender.on_data_sent = _on_data_sent
        tcp_sender.on_status_update = _on_status_update
        tcp_sender.on_retransmission_requested = _on_retx

        def _do_connect():
            try:
                ok = False
                if hasattr(tcp_sender, 'connect'):
                    ok = tcp_sender.connect(ip, port)
                with state_lock:
                    app_state['sender_connected'] = bool(ok)
                    app_state['status'] = f'已连接到 {ip}:{port}' if ok else f'连接到 {ip}:{port} 失败'
                    app_state['status_level'] = 'success' if ok else 'danger'
                    app_state['encoding_method'] = encoding_method
                    app_state['error_rate'] = float(error_rate or 0.0)
                app.logger.info(f'发送端连接结果: {ok}, sender_connected={app_state["sender_connected"]}')
            except Exception as e:
                with state_lock:
                    app_state['sender_connected'] = False
                    app_state['status'] = f'连接异常: {e}'
                    app_state['status_level'] = 'danger'
                app.logger.exception(f'发送端连接异常: {e}')

        threading.Thread(target=_do_connect, daemon=True).start()

        return jsonify({'success': True, 'status': app_state['status'], 'status_level': app_state['status_level']})
    except Exception as e:
        with state_lock:
            app_state['status'] = f'连接失败: {str(e)}'
            app_state['status_level'] = 'danger'
        return jsonify({'success': False, 'error': str(e), 'status': app_state['status'], 'status_level': app_state['status_level']})


@app.route('/api/disconnect/sender', methods=['POST'])
def disconnect_sender():
    global tcp_sender
    try:
        if tcp_sender and hasattr(tcp_sender, 'disconnect'):
            try:
                tcp_sender.disconnect()
            except Exception:
                pass
            tcp_sender = None

        with state_lock:
            app_state['sender_connected'] = False
            app_state['status'] = '发送端已断开连接'
            app_state['status_level'] = 'info'

        return jsonify({'success': True, 'status': app_state['status'], 'status_level': app_state['status_level']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'status': app_state.get('status', ''), 'status_level': app_state.get('status_level', 'danger')})


@app.route('/api/connect/receiver', methods=['POST'])
def connect_receiver():
    global tcp_receiver
    data = request.json
    try:
        port = int(data.get('port') or 0)
    except Exception:
        port = 0
    encoding_method = data.get('encoding_method')

    try:
        tcp_receiver = TCPReceiver()
        tcp_receiver.encoding_method = encoding_method
        _reset_receive_state()

        if hasattr(tcp_receiver, 'on_message_received'):
            tcp_receiver.on_message_received = handle_tcp_message

        # 回调绑定（若 backend 提供这些回调属性则会被调用）
        if hasattr(tcp_receiver, 'on_data_received'):
            def _on_data_received(*args, **kwargs):
                """
                支持多种后端回调签名：
                  - on_data_received(bytes_or_path)
                  - on_data_received(header_dict, payload_bytes)
                自动选择合适的处理函数，避免把 header 当作 bytes 写入造成文件损坏。
                """
                try:
                    if len(args) == 1:
                        handle_received_data(args[0], kwargs.get('filename'))
                        return
                    if len(args) >= 2 and isinstance(args[0], dict):
                        header, payload = args[0], args[1]
                        handle_received_data_from_backend(header, payload)
                        return
                    if args:
                        handle_received_data(args[0], kwargs.get('filename'))
                    else:
                        handle_received_data(kwargs.get('data') or b'', kwargs.get('filename'))
                except Exception:
                    app.logger.exception("on_data_received wrapper error")

            tcp_receiver.on_data_received = _on_data_received

        # 接收完成复位（注意：实际处理在handle_tcp_message中，这里只做备用）
        if hasattr(tcp_receiver, 'on_transmission_complete'):
            def _on_rx_complete():
                # 不在这里处理，让handle_tcp_message统一处理
                pass
            tcp_receiver.on_transmission_complete = _on_rx_complete

        if hasattr(tcp_receiver, 'on_progress'):
             tcp_receiver.on_progress = update_receive_progress

        # 自动接受连接请求（仅用于本机调试/自连）
        if hasattr(tcp_receiver, 'on_connection_request'):
            def _auto_accept(addr, enc_method):
                try:
                    if hasattr(tcp_receiver, 'accept_connection'):
                        tcp_receiver.accept_connection()
                    with state_lock:
                        app_state['status'] = f'收到连接请求并自动接受: {addr}'
                        app_state['status_level'] = 'info'
                        app_state['encoding_method'] = enc_method
                except Exception as e:
                    with state_lock:
                        app_state['status'] = f'自动接受失败: {e}'
                        app_state['status_level'] = 'danger'
            tcp_receiver.on_connection_request = _auto_accept

        if hasattr(tcp_receiver, 'on_connection_established'):
            tcp_receiver.on_connection_established = lambda addr: None

        def find_free_port():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', 0))
            p = s.getsockname()[1]
            s.close()
            return p

        def _start_listen():
            try:
                ok = False
                actual_port = port
                if hasattr(tcp_receiver, 'start_listening'):
                    ok = tcp_receiver.start_listening(port)
                if not ok:
                    try_port = find_free_port()
                    ok = tcp_receiver.start_listening(try_port)
                    if ok:
                        actual_port = try_port

                with state_lock:
                    app_state['receiver_connected'] = bool(ok)
                    app_state['status'] = f'接收端已启动，监听端口 {actual_port}' if ok else f'监听端口 {port} 失败'
                    app_state['status_level'] = 'success' if ok else 'danger'
                    if ok:
                        app_state['listening_port'] = actual_port
            except Exception as e:
                with state_lock:
                    app_state['receiver_connected'] = False
                    app_state['status'] = f'接收异常: {e}'
                    app_state['status_level'] = 'danger'

        threading.Thread(target=_start_listen, daemon=True).start()

        return jsonify({'success': True, 'status': app_state['status'], 'status_level': app_state['status_level']})
    except Exception as e:
        with state_lock:
            app_state['status'] = f'接收端启动失败: {str(e)}'
            app_state['status_level'] = 'danger'
        return jsonify({'success': False, 'error': str(e), 'status': app_state.get('status', ''), 'status_level': app_state.get('status_level', 'danger')})


@app.route('/api/disconnect/receiver', methods=['POST'])
def disconnect_receiver():
    global tcp_receiver
    try:
        if tcp_receiver and hasattr(tcp_receiver, 'stop_listening'):
            try:
                tcp_receiver.stop_listening()
            except Exception:
                pass
            tcp_receiver = None

        with state_lock:
            app_state['receiver_connected'] = False
            app_state['status'] = '接收端已停止'
            app_state['status_level'] = 'info'
        _reset_receive_state()

        return jsonify({'success': True, 'status': app_state['status'], 'status_level': app_state['status_level']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'status': app_state.get('status', ''), 'status_level': app_state.get('status_level', 'danger')})


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有文件部分'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '未选择文件'})

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 获取文件信息（FileProcessor 需提供 get_file_metadata）
        file_info = {}
        try:
            file_info = file_processor.get_file_metadata(filepath)
        except Exception:
            file_info = {}

        with state_lock:
            app_state['last_uploaded_name'] = file.filename

        return jsonify({
            'success': True,
            'filename': file.filename,
            'filepath': filepath,
            'file_info': file_info
        })

    return jsonify({'success': False, 'error': '不支持的文件类型'})


def update_send_progress(progress, bytes_sent, messages_sent, retrans_count):
    with state_lock:
        app_state['progress'] = progress
        app_state['stats']['bytes_sent'] = bytes_sent
        app_state['stats']['messages_sent'] = messages_sent
        app_state['stats']['retrans_count'] = retrans_count


def update_receive_progress(received, total=None):
    try:
        with state_lock:
            percent = None
            if isinstance(received, dict):
                b = received.get('bytes') or received.get('received') or 0
                t = received.get('total') or received.get('size') or total
                if t:
                    percent = int(b * 100 / t)
                else:
                    percent = int(b) if isinstance(b, (int, float)) else 0
            elif total is not None and isinstance(received, (int, float)):
                if total:
                    percent = int(received * 100 / total)
                else:
                    percent = int(received)
            elif isinstance(received, (int, float)):
                percent = int(received)
            else:
                percent = 0

            percent = max(0, min(100, int(percent)))
            app_state['progress'] = percent
            app_state['receiving_in_progress'] = (percent < 100)
            app_state['status'] = '接收中...' if percent < 100 else '接收完成'
            app_state['status_level'] = 'info' if percent < 100 else 'success'
    except Exception as e:
        with state_lock:
            app_state['status'] = f'更新接收进度时出错: {e}'
            app_state['status_level'] = 'danger'


class WebFileSenderWorker(threading.Thread):
    """后台线程：负责编码、信道模拟、CRC 发送等完整流程"""

    def __init__(self, sender: TCPSender, file_path: str, encoding_method: str, error_rate: float,
                 chunk_size: int = 4200):
        super().__init__(daemon=True)
        self.sender = sender
        self.file_path = file_path
        self.encoding_method = encoding_method or 'linear_7_4'
        # 误码率支持0.001精度（千分之一），范围[0.0, 0.5]
        self.error_rate = max(0.0, min(0.5, float(error_rate or 0.0)))
        self.chunk_size = chunk_size
        self._stop_event = threading.Event()
        self.file_crc_value = None
        self.total_chunks = 0
        self.sent_chunks = 0
        self._bsc_channel = BSCChannel(error_probability=self.error_rate)

    def stop(self):
        self._stop_event.set()

    def _emit_status(self, message: str, level: str = 'info'):
        _set_status(message, level)

    def _update_progress(self, progress: int):
        with state_lock:
            app_state['progress'] = max(0, min(100, int(progress)))

    def _calculate_crc(self):
        try:
            with open(self.file_path, 'rb') as f:
                data = f.read()
            self.file_crc_value = crc_calculator.calculate(data)
            return len(data)
        except Exception as exc:
            self._emit_status(f'计算文件 CRC 失败: {exc}', 'danger')
            self.file_crc_value = None
            return 0

    def _send_crc_message(self, file_size: int):
        if not self.sender or not self.sender.is_connected or self.file_crc_value is None:
            return
        metadata = {
            'crc_value': self.file_crc_value,
            'file_size': file_size,
            'file_name': os.path.basename(self.file_path),
            'encoding_method': self.encoding_method,
            'error_rate': self.error_rate,
        }
        try:
            crc_msg = TCPMessage(MessageType.FILE_CRC, data=metadata)
            self.sender._send_message(crc_msg)
        except Exception as exc:
            self._emit_status(f'发送 CRC 消息失败: {exc}', 'warning')

    def _encode_and_simulate(self, chunk: bytes):
        """编码数据并通过BSC信道模拟传输"""
        # 步骤1: 使用选定的编码方法对原始数据进行编码
        original_bit_len = len(chunk) * 8
        original_byte_len = len(chunk)
        
        encoded_bits = channel_encoder.encode(chunk, self.encoding_method)
        encoded_bit_len = int(len(encoded_bits))
        
        # 添加编码调试信息
        app.logger.debug(
            f"编码信息 [编码方法={self.encoding_method}]: "
            f"原始={original_byte_len}字节/{original_bit_len}比特, "
            f"编码后={encoded_bit_len}比特"
        )
        
        # 步骤2: 通过BSC信道模拟，添加误码
        # 如果误码率为0，直接使用编码后的数据，不经过BSC（避免不必要的随机性）
        if self.error_rate == 0.0:
            corrupted_bits = encoded_bits.copy()
            actual_ber = 0.0
            total_errors = 0
        else:
            self._bsc_channel.set_error_probability(self.error_rate)
            corrupted_bits = self._bsc_channel.transmit(encoded_bits)
            stats = self._bsc_channel.get_statistics()
            actual_ber = stats.get('actual_ber', 0.0)
            total_errors = stats.get('total_errors', 0)
        
        # 步骤3: 将误码后的编码比特转换回字节用于传输
        # 注意：确保比特长度是8的倍数，否则bits_to_bytes会填充
        # 关键：记录转换前的编码比特长度，因为转换后可能会添加填充
        payload_bytes = bits_to_bytes(corrupted_bits)
        
        # 计算实际传输的字节数对应的比特数（去除填充）
        # 这很重要，因为bits_to_bytes可能会添加填充，但我们需要知道原始编码比特长度
        actual_encoded_bit_len = len(corrupted_bits)  # 这是BSC信道后的实际编码比特长度
        
        # 记录元数据，供接收端解码使用
        metadata = {
            'encoded': True,
            'encoding_method': self.encoding_method,
            'encoded_bit_len': actual_encoded_bit_len,  # 实际编码比特长度（BSC后，不含字节转换填充）
            'original_bit_len': original_bit_len,  # 原始数据比特长度（不含编码填充）
            'original_byte_len': original_byte_len,  # 原始数据字节长度
            'error_rate': self.error_rate,
            'actual_ber': actual_ber,
            'total_bits': encoded_bit_len,  # 原始编码比特长度（BSC前）
            'total_errors': total_errors,
        }
        
        app.logger.debug(
            f"编码完成 [编码方法={self.encoding_method}]: "
            f"原始编码比特={encoded_bit_len}比特, "
            f"BSC后编码比特={actual_encoded_bit_len}比特, "
            f"传输字节={len(payload_bytes)}字节"
        )
        
        return payload_bytes, metadata

    def run(self):
        try:
            file_size = self._calculate_crc()
            self._send_crc_message(file_size)

            iterator = file_processor.create_iterator(self.file_path, self.chunk_size)
            file_info = iterator.get_file_info()
            self.total_chunks = max(1, int(file_info.get('total_chunks', 1)))

            with state_lock:
                app_state['stats']['bytes_sent'] = 0
                app_state['stats']['messages_sent'] = 0
                app_state['error_rate'] = self.error_rate
                app_state['encoding_method'] = self.encoding_method
                app_state['sending_in_progress'] = True
                app_state['status'] = '正在发送文件...'
                app_state['status_level'] = 'info'

            for chunk in iterator:
                if self._stop_event.is_set():
                    self._emit_status('发送已被取消', 'warning')
                    return
                if not self.sender or not self.sender.is_connected:
                    self._emit_status('连接已断开，发送中止', 'danger')
                    return

                payload_bytes, metadata = self._encode_and_simulate(chunk)
                message = TCPMessage(MessageType.DATA, data=payload_bytes, metadata=metadata)
                self.sender._send_message(message)

                self.sent_chunks += 1
                progress = int(self.sent_chunks * 100 / self.total_chunks)
                self._update_progress(progress)

                with state_lock:
                    app_state['stats']['bytes_sent'] += len(chunk)
                    app_state['stats']['messages_sent'] += 1

                time.sleep(0.001)  # 稍作让步，避免占用 CPU

            if self.sender and self.sender.is_connected:
                complete_msg = TCPMessage(MessageType.TRANSMISSION_COMPLETE, metadata={
                    'encoding_method': self.encoding_method,
                    'file_name': os.path.basename(self.file_path),
                    'crc_value': self.file_crc_value,
                })
                self.sender._send_message(complete_msg)
                self._emit_status('✅ 文件发送完成', 'success')
            else:
                self._emit_status('发送完成但连接已断开', 'warning')

        except Exception as exc:
            self._emit_status(f'发送过程中出错: {exc}', 'danger')
        finally:
            with state_lock:
                app_state['sending_in_progress'] = False
                app_state['progress'] = 100 if self.sent_chunks >= self.total_chunks else app_state.get('progress', 0)
            global sender_worker
            sender_worker = None


def _reset_receive_state():
    with state_lock:
        app_state['current_receive'] = {
            'buffer': bytearray(),
            'expected_crc': None,
            'file_name': None,
            'encoding_method': None,
            'error_rate': None,
            'start_ts': time.time(),
            'last_update': time.time(),
        }


def _append_received_bytes(data: bytes, metadata: Optional[dict] = None):
    if not isinstance(data, (bytes, bytearray)):
        return
    metadata = metadata or {}
    with state_lock:
        recv_state = app_state.setdefault('current_receive', {})
        if not recv_state:
            _reset_receive_state()
            recv_state = app_state['current_receive']
        buffer = recv_state.setdefault('buffer', bytearray())
        buffer.extend(data)
        recv_state['last_update'] = time.time()
        if metadata.get('file_name'):
            recv_state['file_name'] = metadata['file_name']
        if metadata.get('encoding_method'):
            recv_state['encoding_method'] = metadata['encoding_method']
        if metadata.get('error_rate') is not None:
            recv_state['error_rate'] = metadata['error_rate']


def _set_expected_crc(crc_value: Optional[int], file_name: Optional[str], encoding_method: Optional[str],
                      error_rate: Optional[float]):
    with state_lock:
        recv_state = app_state.setdefault('current_receive', {})
        if not recv_state:
            _reset_receive_state()
            recv_state = app_state['current_receive']
        recv_state['expected_crc'] = crc_value
        if file_name:
            recv_state['file_name'] = file_name
        if encoding_method:
            recv_state['encoding_method'] = encoding_method
        if error_rate is not None:
            recv_state['error_rate'] = error_rate
        recv_state['start_ts'] = recv_state.get('start_ts') or time.time()
        recv_state['last_update'] = time.time()


def _request_retransmission(reason: str = ''):
    """由接收端主动向发送端发起重传请求。
    要求接收端与发送端已经建立TCP连接（tcp_receiver.is_connected）。
    """
    global tcp_receiver
    try:
        if not tcp_receiver or not getattr(tcp_receiver, 'is_connected', False):
            _set_status('接收端未连接，无法发送重传请求', 'danger')
            return False
        ok = tcp_receiver.request_retransmission()
        if ok:
            msg = '已向发送端请求重传'
            if reason:
                msg += f'：{reason}'
            _set_status(msg, 'warning')
        else:
            _set_status('重传请求发送失败', 'danger')
        return ok
    except Exception as e:
        _set_status(f'发送重传请求异常: {e}', 'danger')
        return False


def _finalize_receive_data():
    with state_lock:
        recv_state = app_state.get('current_receive') or {}
        buffer = bytes(recv_state.get('buffer') or b'')
        expected_crc = recv_state.get('expected_crc')
        file_name = recv_state.get('file_name')
        # 立即重置接收状态，避免重复处理
        app_state['receiving_in_progress'] = False
        if 'buffer' in app_state.get('current_receive', {}):
            app_state['current_receive']['buffer'] = bytearray()

    if not buffer:
        _set_status('未接收到有效数据', 'warning')
        _reset_receive_state()
        return

    # 计算接收到的数据的CRC
    actual_crc = crc_calculator.calculate(buffer)
    
    # 添加调试信息
    recv_state = app_state.get('current_receive', {})
    encoding_method = recv_state.get('encoding_method', 'unknown')
    # 优先从recv_state获取error_rate，如果没有则从app_state获取，最后默认为0.0
    error_rate = recv_state.get('error_rate')
    if error_rate is None:
        error_rate = app_state.get('error_rate', 0.0)
    else:
        error_rate = float(error_rate)
    
    app.logger.info(
        f"CRC校验: 期望={expected_crc:#010x}, 实际={actual_crc:#010x}, "
        f"数据长度={len(buffer)}字节, 编码方法={encoding_method}, 误码率={error_rate}"
    )
    
    if expected_crc is not None and actual_crc != expected_crc:
        with state_lock:
            app_state['stats']['errors_detected'] = app_state['stats'].get('errors_detected', 0) + 1
            app_state['pending_retrans'] = {
                'reason': f'CRC mismatch expected=0x{expected_crc:08X} actual=0x{actual_crc:08X}',
                'ts': time.time()
            }
        _set_status(
            f'❌ CRC 校验失败 (期望 0x{expected_crc:08X}, 实际 0x{actual_crc:08X}, 数据长度: {len(buffer)}字节)；等待用户确认是否重传',
            'danger'
        )
    else:
        with state_lock:
            app_state['pending_retrans'] = None
        _set_status(f'✅ CRC 校验通过 (数据长度: {len(buffer)}字节)', 'success')

    _write_final_file_and_update_state(buffer, file_name)
    _reset_receive_state()
    
    # 确保接收状态已重置
    with state_lock:
        app_state['receiving_in_progress'] = False
        app_state['progress'] = 0


def _decode_message_payload(message: TCPMessage) -> bytes:
    """解码接收到的消息载荷，处理编码数据的解码流程"""
    data = message.data
    metadata = message.metadata or {}

    if metadata.get('encoded'):
        # 这是经过编码和BSC信道误码的数据，需要解码
        encoding_method = metadata.get('encoding_method') or app_state.get('encoding_method') or 'linear_7_4'
        encoded_bytes = b''
        if isinstance(data, bytes):
            encoded_bytes = data
        elif isinstance(data, bytearray):
            encoded_bytes = bytes(data)
        elif isinstance(data, list):
            encoded_bytes = bytes(data)
        else:
            encoded_bytes = pickle.dumps(data)

        # 将接收到的字节转换回比特数组
        all_encoded_bits = bytes_to_bits(encoded_bytes)
        encoded_bit_len = int(metadata.get('encoded_bit_len', len(all_encoded_bits)))
        original_bit_len = int(metadata.get('original_bit_len', 0))
        original_byte_len = int(metadata.get('original_byte_len', 0))
        
        # 添加接收数据检查
        app.logger.debug(
            f"接收数据检查 [编码方法={encoding_method}]: "
            f"接收字节={len(encoded_bytes)}字节, "
            f"转换后比特={len(all_encoded_bits)}比特, "
            f"期望编码比特={encoded_bit_len}比特, "
            f"原始={original_byte_len}字节/{original_bit_len}比特"
        )
        
        # 关键修复：先根据编码方法确定需要的比特长度（必须是n的倍数）
        # 然后截取到正确的长度，确保解码器能正确处理
        decoder = channel_encoder.encoders.get(encoding_method)
        if not decoder:
            app.logger.error(f"未找到解码器: {encoding_method}")
            return b''
        
        # 确定解码器需要的块大小
        if encoding_method == 'linear_7_4':
            block_size = 7
        elif encoding_method == 'linear_3_2':
            block_size = 3
        elif encoding_method == 'conv_7_4_3':
            block_size = 7
        elif encoding_method == 'conv_2_1_2':
            block_size = 2
        else:
            block_size = 1
        
        # 关键修复：直接使用encoded_bit_len，因为这是发送端计算的准确值
        # 但需要确保是block_size的倍数
        if encoded_bit_len % block_size != 0:
            # 如果不是倍数，向下取整到最近的倍数
            max_valid_bits = (encoded_bit_len // block_size) * block_size
            app.logger.warning(
                f"编码比特长度不是{block_size}的倍数: {encoded_bit_len}, "
                f"调整为: {max_valid_bits}"
            )
        else:
            max_valid_bits = encoded_bit_len
        
        # 从all_encoded_bits中截取，确保不超过实际长度
        if max_valid_bits > len(all_encoded_bits):
            app.logger.error(
                f"编码比特长度超出实际数据: 期望={max_valid_bits}比特, "
                f"实际={len(all_encoded_bits)}比特"
            )
            # 截取到实际长度的最大block_size倍数
            max_valid_bits = (len(all_encoded_bits) // block_size) * block_size
        
        encoded_bits = all_encoded_bits[:max_valid_bits]
        
        # 验证截取后的长度
        if len(encoded_bits) % block_size != 0:
            app.logger.error(
                f"截取后的编码比特长度 {len(encoded_bits)} 不是 {block_size} 的倍数！"
            )
            return b''
        
        app.logger.debug(
            f"准备解码 [编码方法={encoding_method}]: "
            f"编码比特={len(encoded_bits)}比特 (block_size={block_size}), "
            f"原始比特={original_bit_len}比特"
        )
        
        try:
            # 解码
            decoded_bits = decoder.decode(np.array(encoded_bits, dtype=int))
            
            # 添加解码前的详细日志
            app.logger.debug(
                f"解码前检查 [编码方法={encoding_method}]: "
                f"编码比特={len(encoded_bits)}比特, "
                f"原始比特={original_bit_len}比特, "
                f"原始字节={original_byte_len}字节, "
                f"解码后比特={len(decoded_bits)}比特"
            )
                
            # 关键：根据原始长度截取解码后的比特
            # 对于线性分组码：解码后可能包含填充，需要根据original_bit_len截取
            # 对于卷积码：解码器会解码所有组（包括填充），需要根据original_bit_len截取去除填充
            if original_bit_len > 0:
                if len(decoded_bits) > original_bit_len:
                    # 截取到原始长度（去除填充）
                    before_len = len(decoded_bits)
                    decoded_bits = decoded_bits[:original_bit_len]
                    app.logger.debug(
                        f"截取填充 [编码方法={encoding_method}]: "
                        f"从{before_len}比特截取到{original_bit_len}比特"
                    )
                elif len(decoded_bits) < original_bit_len:
                    # 如果解码后长度不足，这是严重问题
                    # 对于线性分组码，这通常意味着解码错误或数据丢失
                    app.logger.error(
                        f"解码后比特长度不足 [编码方法={encoding_method}]: "
                        f"期望={original_bit_len}比特, 实际={len(decoded_bits)}比特, "
                        f"缺失={original_bit_len - len(decoded_bits)}比特, "
                        f"编码比特长度={encoded_bit_len}, "
                        f"截取后编码比特={len(encoded_bits)}, "
                        f"原始字节={original_byte_len}"
                    )
                    # 关键修复：如果长度不足，尝试用0填充到原始长度
                    # 这样可以保持数据完整性，让CRC校验来检测错误
                    padding_needed = original_bit_len - len(decoded_bits)
                    decoded_bits = np.concatenate([decoded_bits, np.zeros(padding_needed, dtype=int)])
                    app.logger.warning(
                        f"已用0填充缺失的{padding_needed}比特，CRC校验将检测错误"
                    )
                # 如果长度正好匹配，不需要处理
            
            # 转换为字节（bits_to_bytes会自动处理填充）
            decoded_bytes = bits_to_bytes(decoded_bits)
            
            # 关键修复：根据原始字节长度精确截取
            # 这是最重要的步骤，确保解码后的字节数完全匹配原始数据
            if original_byte_len > 0:
                if len(decoded_bytes) > original_byte_len:
                    # 截取到原始字节长度（去除bits_to_bytes添加的填充）
                    before_len = len(decoded_bytes)
                    decoded_bytes = decoded_bytes[:original_byte_len]
                    app.logger.debug(
                        f"截取字节填充 [编码方法={encoding_method}]: "
                        f"从{before_len}字节截取到{original_byte_len}字节"
                    )
                elif len(decoded_bytes) < original_byte_len:
                    # 如果解码后长度不足，这是严重错误
                    bytes_needed = original_byte_len - len(decoded_bytes)
                    app.logger.error(
                        f"解码后字节长度不足 [编码方法={encoding_method}]: "
                        f"期望={original_byte_len}字节, 实际={len(decoded_bytes)}字节, "
                        f"缺失={bytes_needed}字节, "
                        f"原始比特={original_bit_len}, "
                        f"解码后比特={len(decoded_bits)}"
                    )
                    # 关键修复：如果解码后字节不足，用0填充到原始长度
                    # 这样可以保持数据完整性，让CRC校验来检测错误
                    padding_bytes = b'\x00' * bytes_needed
                    decoded_bytes = decoded_bytes + padding_bytes
                    app.logger.warning(
                        f"已用0填充缺失的{bytes_needed}字节，CRC校验将检测错误"
                    )
            
            # 添加详细的调试信息
            # 优先从metadata获取，如果没有则从app_state获取，最后默认为0.0
            error_rate = metadata.get('error_rate')
            if error_rate is None:
                error_rate = app_state.get('error_rate', 0.0)
            else:
                error_rate = float(error_rate)
            
            # 始终记录解码信息，特别是当有错误时
            app.logger.info(
                f"解码验证 [误码率={error_rate}, 编码方法={encoding_method}]: "
                f"原始={original_byte_len}字节/{original_bit_len}比特, "
                f"编码={encoded_bit_len}比特, "
                f"截取后编码={len(encoded_bits)}比特, "
                f"解码后={len(decoded_bits)}比特/{len(decoded_bytes)}字节, "
                f"最终={len(decoded_bytes)}字节"
            )
            
            # 如果长度不匹配，记录警告
            if original_byte_len > 0 and len(decoded_bytes) != original_byte_len:
                app.logger.warning(
                    f"长度不匹配警告: 期望={original_byte_len}字节, 实际={len(decoded_bytes)}字节, "
                    f"差异={len(decoded_bytes) - original_byte_len}字节"
                )
            
            return decoded_bytes
        except Exception as exc:
            # 获取误码率用于日志
            log_error_rate = metadata.get('error_rate')
            if log_error_rate is None:
                log_error_rate = app_state.get('error_rate', 0.0)
            app.logger.exception(
                f"解码异常 [编码方法={encoding_method}, 误码率={log_error_rate}]: {exc}"
            )
            # 解码失败时返回空，让上层处理
            return b''

    # 未编码的数据直接返回
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, list):
        return bytes(data)
    return str(data).encode('utf-8', errors='replace')


def handle_tcp_message(message: TCPMessage):
    if not isinstance(message, TCPMessage):
        return

    if message.type == MessageType.FILE_CRC:
        crc_info = message.data if isinstance(message.data, dict) else {}
        crc_value = crc_info.get('crc_value')
        file_name = crc_info.get('file_name')
        encoding_method = crc_info.get('encoding_method')
        error_rate = crc_info.get('error_rate')
        _reset_receive_state()
        _set_expected_crc(crc_value, file_name, encoding_method, error_rate)
        status_text = f"收到文件 CRC: {crc_value:#010x}" if crc_value is not None else "收到文件 CRC 消息"
        _set_status(status_text, 'info')
        return

    if message.type == MessageType.DATA:
        decoded_bytes = _decode_message_payload(message)
        if not decoded_bytes:
            _set_status('解码失败，跳过该数据块', 'warning')
            return
        
        # 检查解码后的数据长度
        metadata = message.metadata or {}
        original_byte_len = metadata.get('original_byte_len', 0)
        if original_byte_len > 0 and len(decoded_bytes) < original_byte_len:
            app.logger.error(
                f"数据块不完整: 期望={original_byte_len}字节, "
                f"实际={len(decoded_bytes)}字节, "
                f"缺失={original_byte_len - len(decoded_bytes)}字节"
            )
            # 即使长度不足，也继续处理，让CRC校验来检测错误
        
        _append_received_bytes(decoded_bytes, metadata)
        with state_lock:
            app_state['stats']['bytes_received'] = app_state['stats'].get('bytes_received', 0) + len(decoded_bytes)
            app_state['stats']['messages_received'] = app_state['stats'].get('messages_received', 0) + 1
            app_state['receiving_in_progress'] = True
        # 不频繁更新状态，避免刷屏
        return

    if message.type == MessageType.TRANSMISSION_COMPLETE:
        _set_status('收到传输完成信号，开始 CRC 校验...', 'info')
        _finalize_receive_data()
        return


_reset_receive_state()


def _detect_extension_from_bytes(b: bytes) -> str:
    try:
        t = _img_type_bytes(b)
        if t:
            if t == 'jpeg':
                return '.jpg'
            return f'.{t}'
    except Exception:
        pass

    try:
        if len(b) >= 12 and b[4:8] == b'ftyp':
            return '.mp4'
        if b'ftyp' in b[:256]:
            return '.mp4'
    except Exception:
        pass

    # 不再默认按文本写入，避免错误判断导致破损
    return ''


def _write_final_file_and_update_state(bytes_data: bytes, prefer_name: Optional[str]):
    downloads = app.config['DOWNLOAD_FOLDER']
    os.makedirs(downloads, exist_ok=True)

    base_name = (prefer_name or f"received_{int(time.time())}")
    base_name = os.path.basename(base_name)
    name_root, ext = os.path.splitext(base_name)
    if not ext:
        guessed = _detect_extension_from_bytes(bytes_data)
        if guessed:
            ext = guessed
    final_name = f"{name_root}{ext}"

    target = os.path.join(downloads, final_name)
    idx = 1
    while os.path.exists(target):
        target = os.path.join(downloads, f"{name_root}_{idx}{ext}")
        idx += 1

    tmp = target + '.tmp'
    try:
        with open(tmp, 'wb') as fw:
            fw.write(bytes_data)
        os.replace(tmp, target)

        file_hash = hashlib.sha256(bytes_data).hexdigest()

        with state_lock:
            rh = app_state.setdefault('received_hashes', [])
            if file_hash in rh:
                # 内容重复：不再删除新文件，保留副本并继续记录
                app.logger.info(f"检测到重复内容，仍保留副本: {target}")
            app_state['received_files'].insert(0, {'filename': os.path.basename(target)})
            app_state['stats']['bytes_received'] = app_state['stats'].get('bytes_received', 0) + len(bytes_data)
            app_state['stats']['messages_received'] = app_state['stats'].get('messages_received', 0) + 1
            rh.insert(0, file_hash)
            if len(rh) > 500:
                rh.pop()
            app_state['progress'] = 100
            app_state['receiving_in_progress'] = False
            app_state['status'] = f'文件接收完成: {os.path.basename(target)}'
            app_state['status_level'] = 'success'
            perf = app_state.setdefault('performance_history', [])
            perf.append({
                'time': time.strftime('%H:%M:%S'),
                'speed': round(len(bytes_data)/1024, 2),
                'error_rate': app_state['stats'].get('errors_detected', 0)
            })
            if len(perf) > 50:
                perf.pop(0)
        app.logger.debug(f"写入完成: {target} size={len(bytes_data)} sha256={file_hash}")
        return True
    except Exception as e:
        app.logger.exception("写文件失败")
        with state_lock:
            app_state['status'] = f'写文件失败: {e}'
            app_state['status_level'] = 'danger'
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


def _ephemeral_cleanup_worker():
    """
    周期性 flush ephemeral buffer 并清理超时分片：
      - 仅按“静默超时 > 2s”触发一次性落盘
      - 取消“>2MB 自动落盘”，避免大文件被截断
    """
    while True:
        try:
            now = time.time()
            with state_lock:
                ep = app_state.get('ephemeral_buffers', {}).get('current')
            if ep and (now - ep.get('last_update', ep.get('first_seen', 0)) > 2.0):
                bufdata = bytes(ep.get('buf', b'') if not isinstance(ep.get('buf'), (bytes, bytearray)) else ep.get('buf'))
                prefer_name = ep.get('filename') or app_state.get('sending_filename') or app_state.get('last_uploaded_name')
                ok = False
                try:
                    ok = _write_final_file_and_update_state(bufdata, prefer_name)
                except Exception:
                    app.logger.exception("flush 写文件异常")
                    ok = False
                with state_lock:
                    app_state['ephemeral_buffers']['current'] = None
                if ok:
                    app.logger.debug("ephemeral buffer flush completed")
            with state_lock:
                im = app_state.get('incomplete_messages', {})
                stale = [mid for mid, info in im.items() if info.get('last_update', 0) < (now - 300)]
                for mid in stale:
                    app.logger.debug(f"清理过期分片 {mid}")
                    im.pop(mid, None)
            time.sleep(1.0)
        except Exception:
            app.logger.exception("ephemeral_cleanup 异常")
            time.sleep(1.0)


def handle_received_data_from_backend(header, payload):
    try:
        payload = bytes(payload)
        msg_id = header.get('message_id')
        fname = header.get('filename')
        total_parts = header.get('total_parts')
        part_index = header.get('part_index')
        sha256_total = header.get('sha256_total')

        downloads = app.config['DOWNLOAD_FOLDER']
        os.makedirs(downloads, exist_ok=True)

        if msg_id:
            with state_lock:
                im = app_state.setdefault('incomplete_messages', {})
                info = im.get(msg_id)
                if not info:
                    info = {
                        'parts': {},
                        'total_parts': total_parts,
                        'first_seen': time.time(),
                        'last_update': time.time(),
                        'filename': fname,
                        'sha256_total': sha256_total
                    }
                    im[msg_id] = info
                else:
                    info['last_update'] = time.time()
                    if total_parts and not info.get('total_parts'):
                        info['total_parts'] = total_parts
                    if sha256_total and not info.get('sha256_total'):
                        info['sha256_total'] = sha256_total

                if part_index not in info['parts']:
                    info['parts'][part_index] = payload

                received = len(info['parts'])
                expected = info.get('total_parts')

            if expected is None or received < expected:
                _set_status(f"收到分片 {msg_id} [{received}/{expected if expected else '?'}]", 'info')
                return

            with state_lock:
                info = app_state['incomplete_messages'].get(msg_id)
                parts = info['parts']
                prefer_name = info.get('filename') or fname or app_state.get('sending_filename') or app_state.get('last_uploaded_name') or f"received_{int(time.time())}"
                expected = info.get('total_parts')

            indices = sorted(parts.keys())
            if len(indices) != expected:
                _set_status(f"分片数量不符，丢弃文件 {msg_id}", 'danger')
                with state_lock:
                    app_state['incomplete_messages'].pop(msg_id, None)
                return

            assembled = bytearray()
            for i in indices:
                assembled.extend(parts[i])

            if info.get('sha256_total'):
                calc = hashlib.sha256(assembled).hexdigest()
                if calc != info.get('sha256_total'):
                    app.logger.warning(f"整体 sha256 不匹配 {msg_id} calc={calc} expected={info.get('sha256_total')}")
                    _set_status(f'文件校验失败: {msg_id}', 'danger')
                    with state_lock:
                        app_state['incomplete_messages'].pop(msg_id, None)
                    return

            ok = _write_final_file_and_update_state(bytes(assembled), prefer_name)
            with state_lock:
                app_state['incomplete_messages'].pop(msg_id, None)
            if ok:
                app.logger.debug(f"分片 message_id={msg_id} 写入完成")
            return

        # 无头部情况：合并到 ephemeral 缓冲，依靠“静默 2s”一次性落盘，文件名优先用 sending_filename/last_uploaded_name
        with state_lock:
            cur = app_state['ephemeral_buffers'].get('current')
            if not cur:
                cur = {
                    'buf': bytearray(),
                    'first_seen': time.time(),
                    'last_update': time.time(),
                    'filename': fname or app_state.get('sending_filename') or app_state.get('last_uploaded_name')
                }
                app_state['ephemeral_buffers']['current'] = cur
            cur['buf'].extend(payload)
            cur['last_update'] = time.time()
            _set_status('收到碎片，等待合并...', 'info')

    except Exception:
        app.logger.exception("handle_received_data_from_backend 异常")
        _set_status('文件处理错误', 'danger')


def handle_received_data(data, filename=None):
    """
    统一处理接收到的数据并保存到 app.config['DOWNLOAD_FOLDER']。
    支持：
      - data 为 bytes/bytearray -> 直接写入文件
      - data 为后端提供的文件路径字符串 -> 复制到下载目录
      - 其它类型 -> 转为字符串写入（降级处理）
    """
    try:
        downloads = app.config.get('DOWNLOAD_FOLDER') or os.path.expanduser('~/Downloads')
        os.makedirs(downloads, exist_ok=True)

        if not filename:
            with state_lock:
                filename = app_state.get('sending_filename') or app_state.get('last_uploaded_name') or f"received_{int(time.time())}"

        filename = os.path.basename(filename)
        dest_path = os.path.join(downloads, filename)

        if isinstance(data, (bytes, bytearray)):
            with open(dest_path, 'wb') as fw:
                fw.write(data)
        elif isinstance(data, str) and os.path.exists(data):
            shutil.copy2(data, dest_path)
        else:
            with open(dest_path, 'wb') as fw:
                fw.write(str(data).encode('utf-8'))

        with state_lock:
            app_state['receiving_in_progress'] = False
            app_state['progress'] = 100
            app_state['status'] = f'已保存接收文件: {dest_path}'
            app_state['status_level'] = 'success'
    except Exception as e:
        with state_lock:
            app_state['receiving_in_progress'] = False
            app_state['status'] = f'保存接收文件失败: {e}'
            app_state['status_level'] = 'danger'


@app.route('/api/send', methods=['POST'])
def api_send():
    """
    前端 POST JSON { "filepath": "C:\\path\\to\\file" }
    后台线程调用 tcp_sender 的发送接口（优先 send_file，其次 send_bytes）。
    """
    global tcp_sender, sender_worker
    data = request.get_json(force=True) or {}
    filepath = data.get('filepath')
    encoding_method = data.get('encoding_method') or app_state.get('encoding_method') or 'linear_7_4'
    # 误码率支持0.001精度（千分之一），范围[0.0, 0.5]
    error_rate = float(data.get('error_rate', app_state.get('error_rate', 0.0)))
    # 验证误码率范围
    error_rate = max(0.0, min(0.5, error_rate))
    chunk_size = int(data.get('chunk_size', 4200))

    if not filepath or not os.path.exists(filepath):
        return jsonify({'success': False, 'error': '文件不存在'}), 400

    if not tcp_sender or not tcp_sender.is_connected:
        return jsonify({'success': False, 'error': '未初始化发送端或未连接'}), 500

    if sender_worker and sender_worker.is_alive():
        return jsonify({'success': False, 'error': '已有发送任务正在进行'}), 409

    with state_lock:
        app_state['sending_in_progress'] = True
        app_state['progress'] = 0
        app_state['status'] = '准备发送文件...'
        app_state['status_level'] = 'info'
        app_state['sending_filename'] = os.path.basename(filepath)
        app_state['sending_filepath'] = os.path.abspath(filepath)
        app_state['encoding_method'] = encoding_method
        app_state['error_rate'] = error_rate

    sender_worker = WebFileSenderWorker(
        sender=tcp_sender,
        file_path=filepath,
        encoding_method=encoding_method,
        error_rate=error_rate,
        chunk_size=chunk_size
    )
    sender_worker.start()

    return jsonify({'success': True, 'status': '发送任务已启动'})


@app.route('/api/cancel_send', methods=['POST'])
def cancel_send():
    """取消发送请求的通用接口。"""
    global tcp_sender, sender_worker
    try:
        if sender_worker and sender_worker.is_alive():
            sender_worker.stop()
            sender_worker = None

        if tcp_sender and hasattr(tcp_sender, 'cancel_send'):
            try:
                tcp_sender.cancel_send()
            except Exception:
                pass
        elif tcp_sender and hasattr(tcp_sender, 'cancel'):
            try:
                tcp_sender.cancel()
            except Exception:
                pass
        with state_lock:
            app_state['sending_in_progress'] = False
            app_state['status'] = '发送已取消'
            app_state['status_level'] = 'warning'
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def _set_status(message: str, level: str = 'info'):
    try:
        timestamp = time.strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] [{level.upper()}] {message}"
        
        with state_lock:
            app_state['status'] = str(message)
            app_state['status_level'] = str(level)
            app_state['last_status_time'] = time.time()
            # 添加到日志历史
            logs = app_state.setdefault('status_logs', [])
            logs.append(log_entry)
            # 只保留最近100条日志
            if len(logs) > 100:
                logs.pop(0)
        
        try:
            if level in ('danger', 'warning'):
                app.logger.warning(f"[{level}] {message}")
            else:
                app.logger.info(f"[{level}] {message}")
        except Exception:
            pass
    except Exception:
        try:
            app.logger.exception("_set_status 更新状态时出错")
        except Exception:
            pass


@app.route('/api/request_retransmission', methods=['POST'])
def api_request_retransmission():
    try:
        data = request.get_json(silent=True) or {}
        reason = data.get('reason') or ''
        ok = _request_retransmission(reason=reason)
        # 无论成功与否，清除待确认状态，避免重复弹窗
        with state_lock:
            app_state['pending_retrans'] = None
        return jsonify({'success': bool(ok)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/downloads/<path:filename>')
def download_file(filename):
    try:
        return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)
    except Exception:
        return jsonify({'success': False, 'error': '文件未找到'}), 404


# 在 APP 启动时（模块级）启动该线程一次
if not globals().get('_ephemeral_cleanup_started'):
    t = threading.Thread(target=_ephemeral_cleanup_worker, daemon=True)
    t.start()
    globals()['_ephemeral_cleanup_started'] = True

if __name__ == '__main__':
    # 使用多线程并关闭自动重载，避免启动慢、重复启动以及长请求阻塞
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)