import http.server
import socketserver
import webbrowser
import os
import threading
import socket

class MyTCPServer(socketserver.TCPServer):
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)

def find_free_port(start_port=8000, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# HTML文件在display目录中
HTML_FILE = 'viterbi_simulator.html'
FULL_HTML_PATH = os.path.join(SCRIPT_DIR, HTML_FILE)

# 检查文件是否存在
if not os.path.exists(FULL_HTML_PATH):
    print(f"Error: HTML file not found at {FULL_HTML_PATH}")
    print("Please make sure the file exists and the path is correct.")
    exit(1)

PORT = find_free_port()
if PORT is None:
    print("Could not find an available port")
    exit(1)

# 设置服务器根目录为final目录
os.chdir(SCRIPT_DIR)

# URL应该只包含相对路径部分
URL = f"http://localhost:{PORT}/{HTML_FILE}"

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # 可选：自定义日志输出
        pass

print(f"Server root: {SCRIPT_DIR}")
print(f"HTML file: {FULL_HTML_PATH}")
print(f"Serving at: http://localhost:{PORT}")
print(f"Launching browser to: {URL}")

try:
    httpd = MyTCPServer(("", PORT), Handler)
    
    threading.Timer(1, lambda: webbrowser.open_new_tab(URL)).start()
    
    try:
        print("Server started successfully. Press Ctrl+C to stop.")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.shutdown()
        
except Exception as e:
    print(f"Failed to start server: {e}")
