"""
TCP通信模块
实现发送端和接收端的TCP通信功能
"""

import socket
import threading
import time
import pickle
import struct
from typing import Optional, Callable, Any
from enum import Enum
import os
import hashlib


class MessageType(Enum):
    """消息类型枚举"""
    CONNECTION_REQUEST = "CONNECTION_REQUEST"
    CONNECTION_ACCEPT = "CONNECTION_ACCEPT"
    CONNECTION_REJECT = "CONNECTION_REJECT"
    DATA = "DATA"
    ENCODING_METHOD = "ENCODING_METHOD"
    RETRANSMISSION_REQUEST = "RETRANSMISSION_REQUEST"
    TRANSMISSION_COMPLETE = "TRANSMISSION_COMPLETE"
    STATUS_UPDATE = "STATUS_UPDATE"
    FILE_CRC = "FILE_CRC"


class TCPMessage:
    """TCP消息类"""
    
    def __init__(self, msg_type: MessageType, data: Any = None, metadata: dict = None):
        self.type = msg_type
        self.data = data
        self.metadata = metadata or {}
        self.timestamp = time.time()

    # 修改后的 tcp_communication.py 中 TCPMessage 类的 serialize 方法
    def serialize(self) -> bytes:
        meta = {
            "type": self.type.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
        meta_bytes = pickle.dumps(meta)
        meta_len = struct.pack('!I', len(meta_bytes))

        # 关键修复：将 data 统一序列化为字节（无论原始类型是 dict、str 还是 bytes）
        if self.data is not None:
            # 如果是 bytes 直接使用，否则用 pickle 序列化
            data_bytes = self.data if isinstance(self.data, bytes) else pickle.dumps(self.data)
        else:
            data_bytes = b''

        data_len = struct.pack('!I', len(data_bytes))
        return meta_len + meta_bytes + data_len + data_bytes

    # 修改后的 tcp_communication.py 中 TCPMessage 类的 deserialize 方法
    @classmethod
    def deserialize(cls, data: bytes) -> 'TCPMessage':
        meta_len = struct.unpack('!I', data[:4])[0]
        meta_bytes = data[4:4 + meta_len]
        meta = pickle.loads(meta_bytes)

        data_len_start = 4 + meta_len
        data_len = struct.unpack('!I', data[data_len_start:data_len_start + 4])[0]
        data_bytes = data[data_len_start + 4: data_len_start + 4 + data_len]

        # 尝试反序列化 data（如果是 pickle 序列化的对象）
        try:
            # 先尝试反序列化（适用于 dict 等类型）
            data_content = pickle.loads(data_bytes) if data_bytes else None
        except:
            # 如果反序列化失败，直接使用原始字节（适用于纯二进制数据）
            data_content = data_bytes

        return cls(
            msg_type=MessageType(meta["type"]),
            data=data_content,
            metadata=meta["metadata"]
        )


class TCPSender:
    """TCP发送端类"""
    
    def __init__(self, encoding_method: str = "linear_7_4"):
        """
        初始化TCP发送端
        Args:
            encoding_method: 编码方法
        """
        self.encoding_method = encoding_method
        self.socket = None
        self.connection = None
        self.is_connected = False
        self.receive_thread = None
        self.running = False
        
        # 回调函数
        self.on_status_update = None
        self.on_connection_established = None
        self.on_connection_failed = None
        self.on_data_sent = None
        self.on_retransmission_requested = None
        
        # 统计信息
        self.bytes_sent = 0
        self.messages_sent = 0
        self.retransmission_count = 0
    
    def connect(self, receiver_ip: str, port: int = 5000) -> bool:
        """
        连接到接收端
        Args:
            receiver_ip: 接收端IP地址
            port: 端口号
        Returns:
            连接是否成功
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(30.0)  # 30秒超时
            
            # 发送连接请求
            self._update_status(f"正在连接到 {receiver_ip}:{port}...")
            self.socket.connect((receiver_ip, port))
            self.connection = self.socket
            # 发送连接请求消息
            connect_msg = TCPMessage(MessageType.CONNECTION_REQUEST, 
                                   data={"encoding_method": self.encoding_method})
            self._send_message(connect_msg)
            
            # 等待连接响应
            response = self._receive_message()
            if response and response.type == MessageType.CONNECTION_ACCEPT:
                self.is_connected = True
                self.connection = self.socket
                self.connection.settimeout(None)
                # 发送编码方法信息
                encoding_msg = TCPMessage(MessageType.ENCODING_METHOD, 
                                        data={"method": self.encoding_method})
                self._send_message(encoding_msg)
                
                self._update_status(f"成功建立连接到 {receiver_ip}:{port}")
                
                if self.on_connection_established:
                    self.on_connection_established(receiver_ip, port)
                
                # 启动接收线程
                self.running = True
                self.receive_thread = threading.Thread(target=self._receive_loop)
                self.receive_thread.daemon = True
                self.receive_thread.start()
                
                return True
            else:
                self._update_status("连接被拒绝")
                if self.on_connection_failed:
                    self.on_connection_failed("连接被拒绝")
                return False
                
        except Exception as e:
            self._update_status(f"连接失败: {str(e)}")
            if self.on_connection_failed:
                self.on_connection_failed(str(e))
            return False
    
    def send_data(self, data: bytes) -> bool:
        """
        发送数据
        Args:
            data: 要发送的数据
        Returns:
            发送是否成功
        """
        if not self.is_connected:
            self._update_status("未连接，无法发送数据")
            return False
        
        try:
            # 创建数据消息
            data_msg = TCPMessage(MessageType.DATA, data=data)
            self._send_message(data_msg)
            
            # 更新统计信息
            self.bytes_sent += len(data)
            self.messages_sent += 1
            
            self._update_status(f"已发送 {len(data)} 字节数据")
            
            if self.on_data_sent:
                self.on_data_sent(len(data))
            
            return True
            
        except Exception as e:
            self._update_status(f"发送数据失败: {str(e)}")
            return False

    def send_file(self, file_path: str, chunk_size: int = 4096):
        """
        分片发送文件，给每个分片添加必要的元数据，便于接收端有序重组与校验。
        元数据:
          - message_id: 本次发送的唯一ID
          - filename: 原始文件名
          - part_index: 当前分片序号，从0开始
          - total_parts: 总分片数
          - sha256_total: 整个文件的sha256，用于最后完整性校验
        """
        # 计算整体 sha256
        sha256 = hashlib.sha256()
        total_size = 0
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                sha256.update(chunk)
                total_size += len(chunk)
        sha256_hex = sha256.hexdigest()

        # 基于总大小与分片尺寸计算总分片数
        total_parts = (total_size + chunk_size - 1) // chunk_size
        message_id = f"{int(time.time()*1000)}"
        filename = os.path.basename(file_path)

        # 逐块发送
        with open(file_path, 'rb') as f:
            idx = 0
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                meta = {
                    "message_id": message_id,
                    "filename": filename,
                    "part_index": idx,
                    "total_parts": total_parts,
                    "sha256_total": sha256_hex,
                }
                data_msg = TCPMessage(MessageType.DATA, data=chunk, metadata=meta)
                self._send_message(data_msg)
                # 更新统计
                self.bytes_sent += len(chunk)
                self.messages_sent += 1
                if self.on_data_sent:
                    self.on_data_sent(len(chunk))
                idx += 1

        self.send_completion_signal()
    
    def send_completion_signal(self):
        """发送传输完成信号"""
        if self.is_connected:
            complete_msg = TCPMessage(MessageType.TRANSMISSION_COMPLETE)
            self._send_message(complete_msg)
            self._update_status("发送传输完成信号")
    
    def disconnect(self):
        """断开连接"""
        self.running = False
        self.is_connected = False
        
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=1.0)
        
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
            self.connection = None
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        self._update_status("已断开连接")
    
    def _send_message(self, message: TCPMessage):
        """发送消息"""
        if not self.connection:
            raise RuntimeError("未建立连接")
        
        serialized = message.serialize()
        # 添加消息长度前缀
        message_length = struct.pack('!I', len(serialized))
        self.connection.sendall(message_length + serialized)
    
    def _receive_message(self) -> Optional[TCPMessage]:
        """接收单个消息"""
        if not self.connection:
            return None
        
        try:
            # 接收消息长度
            length_data = self._recv_all(4)
            if not length_data:
                return None
            
            message_length = struct.unpack('!I', length_data)[0]
            
            # 接收消息内容
            message_data = self._recv_all(message_length)
            if not message_data:
                return None
            
            return TCPMessage.deserialize(message_data)
            
        except Exception as e:
            self._update_status(f"接收消息错误: {str(e)}")
            return None
    
    def _recv_all(self, n: int) -> Optional[bytes]:
        """接收指定长度的数据"""
        data = b''
        while len(data) < n:
            packet = self.connection.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _receive_loop(self):
        """接收循环"""
        while self.running and self.is_connected:
            try:
                message = self._receive_message()
                if message:
                    # 单独捕获消息处理时的错误
                    try:
                        self._handle_message(message)
                    except TypeError as e:
                        self._update_status(f"数据处理类型错误: {str(e)}，跳过该消息")
                    except Exception as e:
                        self._update_status(f"数据处理错误: {str(e)}，跳过该消息")
                else:
                    break
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self._update_status(f"接收循环错误: {str(e)}，尝试继续运行")
                # 不直接break，尝试继续接收后续数据

    def _handle_message(self, message: TCPMessage):
        """处理接收到的消息"""
        if message.type == MessageType.RETRANSMISSION_REQUEST:
            self.retransmission_count += 1
            self._update_status("收到重传请求")
            if self.on_retransmission_requested:
                self.on_retransmission_requested()
        
        elif message.type == MessageType.STATUS_UPDATE:
            self._update_status(f"接收端状态: {message.data}")
    
    def _update_status(self, message: str):
        """更新状态"""
        if self.on_status_update:
            self.on_status_update(message)
    
    def get_statistics(self) -> dict:
        """获取发送端统计信息"""
        return {
            'bytes_sent': self.bytes_sent,
            'messages_sent': self.messages_sent,
            'retransmission_count': self.retransmission_count,
            'is_connected': self.is_connected,
            'encoding_method': self.encoding_method
        }


class TCPReceiver:
    """TCP接收端类"""
    
    def __init__(self):
        """初始化TCP接收端"""
        self.server_socket = None
        self.client_connection = None
        self.client_address = None
        self.is_connected = False
        self.is_listening = False
        
        # 接收到的数据
        self.received_data = []
        self.encoding_method = None
        
        # 回调函数
        self.on_connection_request = None
        self.on_connection_established = None
        self.on_data_received = None
        self.on_status_update = None
        self.on_transmission_complete = None
        self.on_message_received = None
        # 线程
        self.listen_thread = None
        self.receive_thread = None
        self.running = False
        
        # 统计信息
        self.bytes_received = 0
        self.messages_received = 0

    def start_listening(self, port: int = 5000) -> bool:
        """
        开始监听连接
        Args:
            port: 监听端口
        Returns:
            是否成功开始监听
        """
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('', port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0)  # 1秒超时
            
            self.is_listening = True
            self.running = True
            
            self.listen_thread = threading.Thread(target=self._listen_loop)
            self.listen_thread.daemon = True
            self.listen_thread.start()
            
            self._update_status(f"正在端口 {port} 上监听连接...")
            return True
            
        except Exception as e:
            self._update_status(f"开始监听失败: {str(e)}")
            return False
    
    def stop_listening(self):
        """停止监听"""
        self.running = False
        self.is_listening = False
        
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=1.0)
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None
        
        self._update_status("已停止监听")
    
    def accept_connection(self) -> bool:
        """
        接受连接请求
        Returns:
            是否成功接受连接
        """
        if not self.client_connection:
            return False
        
        try:
            # 发送接受连接消息
            accept_msg = TCPMessage(MessageType.CONNECTION_ACCEPT)
            self._send_message(accept_msg)
            
            self.is_connected = True
            
            # 启动接收线程
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            self._update_status(f"已接受来自 {self.client_address} 的连接")
            
            if self.on_connection_established:
                self.on_connection_established(self.client_address)
            
            return True
            
        except Exception as e:
            self._update_status(f"接受连接失败: {str(e)}")
            return False
    
    def reject_connection(self, reason: str = "连接被拒绝"):
        """
        拒绝连接请求
        Args:
            reason: 拒绝原因
        """
        if self.client_connection:
            try:
                reject_msg = TCPMessage(MessageType.CONNECTION_REJECT, data=reason)
                self._send_message(reject_msg)
                self.client_connection.close()
            except:
                pass
            finally:
                self.client_connection = None
                self.client_address = None
    
    def request_retransmission(self) -> bool:
        """
        请求重传
        Returns:
            请求是否成功发送
        """
        if not self.is_connected:
            return False
        
        try:
            retrans_msg = TCPMessage(MessageType.RETRANSMISSION_REQUEST)
            self._send_message(retrans_msg)
            self._update_status("已发送重传请求")
            return True
            
        except Exception as e:
            self._update_status(f"发送重传请求失败: {str(e)}")
            return False

    # 在 tcp_communication.py 中修改 _listen_loop 方法

    def _listen_loop(self):
        """监听循环"""
        while self.running and self.is_listening:
            try:
                # 接受连接
                connection, address = self.server_socket.accept()

                self.client_connection = connection
                self.client_address = address

                self._update_status(f"收到来自 {address} 的连接请求")

                # 接收连接请求消息
                message = self._receive_message()
                if message and message.type == MessageType.CONNECTION_REQUEST:
                    # 从消息中获取编码方法
                    encoding_method = message.data.get("encoding_method", "linear_7_4")
                    self.encoding_method = encoding_method  # 保存到实例变量

                    print(f"DEBUG: 接收到编码方法: {encoding_method}")  # 调试信息

                    if self.on_connection_request:
                        # 传递从消息中获取的编码方法
                        self.on_connection_request(f"{address[0]}:{address[1]}", encoding_method)

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self._update_status(f"监听循环错误: {str(e)}")
                break
    
    def _receive_loop(self):
        """接收循环"""
        while self.running and self.is_connected:
            try:
                message = self._receive_message()
                if message:
                    self._handle_message(message)
                else:
                    break
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self._update_status(f"接收循环错误: {str(e)}")
                break
        
        self.is_connected = False
        self._update_status("客户端断开连接")
    
    def _receive_message(self) -> Optional[TCPMessage]:
        """接收单个消息"""
        if not self.client_connection:
            return None
        
        try:
            # 接收消息长度
            length_data = self._recv_all(4)
            if not length_data:
                return None
            
            message_length = struct.unpack('!I', length_data)[0]
            
            # 接收消息内容
            message_data = self._recv_all(message_length)
            if not message_data:
                return None
            
            return TCPMessage.deserialize(message_data)
            
        except Exception as e:
            self._update_status(f"接收消息错误: {str(e)}")
            return None

    def _recv_all(self, n: int) -> Optional[bytes]:
        """接收指定长度的数据"""
        data = b''
        while len(data) < n:
            packet = self.client_connection.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    def _send_message(self, message: TCPMessage):
        """发送消息"""
        if not self.client_connection:
            raise RuntimeError("未建立连接")
        
        serialized = message.serialize()
        message_length = struct.pack('!I', len(serialized))
        self.client_connection.sendall(message_length + serialized)

    def _handle_message(self, message: TCPMessage):
        """处理接收到的消息"""
        if self.on_message_received:
            # 优先使用消息回调
            try:
                self.on_message_received(message)
                return
            except Exception as e:
                print(f"消息回调处理错误: {e}")
                # 如果消息回调出错，回退到原来的处理方式

        # 原有的消息处理逻辑
        if message.type == MessageType.DATA:
            # 确保 payload 为 bytes
            if isinstance(message.data, bytes):
                data_bytes = message.data
            elif isinstance(message.data, list):
                data_bytes = bytes(message.data)
            else:
                # 非法数据类型，丢弃
                self._update_status("收到非二进制DATA，已丢弃")
                return

            self.received_data.append(data_bytes)
            self.bytes_received += len(data_bytes)
            self.messages_received += 1

            self._update_status(f"收到 {len(data_bytes)} 字节数据")

            if self.on_data_received:
                # 若附带分片元数据，则按 (header, payload) 方式回调，匹配前端 APP.py 的处理
                header = {}
                if isinstance(message.metadata, dict):
                    header = {
                        k: message.metadata.get(k)
                        for k in ("message_id", "filename", "part_index", "total_parts", "sha256_total")
                        if k in message.metadata
                    }
                if header:
                    self.on_data_received(header, data_bytes)
                else:
                    # 无头部则按单参数回调，走 APP 的 ephemeral 合并
                    self.on_data_received(data_bytes)

        elif message.type == MessageType.TRANSMISSION_COMPLETE:
            self._update_status("收到传输完成信号")
            if self.on_transmission_complete:
                self.on_transmission_complete()

        elif message.type == MessageType.FILE_CRC:
            # 不再注入到数据流，若有专用回调（如 on_message_received）可由上层消费
            self._update_status("收到文件CRC消息（已分离处理）")

    def _update_status(self, message: str):
        """更新状态"""
        if self.on_status_update:
            self.on_status_update(message)

    def get_received_data(self) -> bytes:
        """获取所有接收到的数据，确保均为bytes类型"""
        # 过滤并转换非bytes类型元素（容错处理）
        valid_data = []
        for item in self.received_data:
            if isinstance(item, bytes):
                valid_data.append(item)
            elif isinstance(item, list):
                valid_data.append(bytes(item))
            else:
                valid_data.append(str(item).encode())
        return b''.join(valid_data)

    def clear_received_data(self):
        """清空接收到的数据"""
        self.received_data.clear()
    
    def get_statistics(self) -> dict:
        """获取接收端统计信息"""
        return {
            'bytes_received': self.bytes_received,
            'messages_received': self.messages_received,
            'is_connected': self.is_connected,
            'is_listening': self.is_listening,
            'encoding_method': self.encoding_method,
            'client_address': self.client_address
        }


class CommunicationManager:
    """通信管理器"""
    
    def __init__(self):
        self.sender = None
        self.receiver = None
    
    def create_sender(self, encoding_method: str = "linear_7_4") -> TCPSender:
        """创建发送端"""
        self.sender = TCPSender(encoding_method)
        return self.sender
    
    def create_receiver(self) -> TCPReceiver:
        """创建接收端"""
        self.receiver = TCPReceiver()
        return self.receiver
    
    def get_statistics(self) -> dict:
        """获取通信统计信息"""
        stats = {
            'sender': self.sender.get_statistics() if self.sender else None,
            'receiver': self.receiver.get_statistics() if self.receiver else None
        }
        return stats