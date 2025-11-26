"""
文件处理模块
支持多种文件类型的迭代处理，防止内存溢出
"""

import os
import struct
from typing import Iterator, Tuple, Optional, BinaryIO
from pathlib import Path


class FileIterator:
    """文件迭代器基类"""
    
    def __init__(self, file_path: str, chunk_size: int = 4200):
        """
        初始化文件迭代器
        Args:
            file_path: 文件路径
            chunk_size: 每次迭代的块大小（字节）
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.file_size = self.file_path.stat().st_size
        self.file_type = self._detect_file_type()
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
    
    def _detect_file_type(self) -> str:
        """检测文件类型"""
        suffix = self.file_path.suffix.lower()
        
        if suffix in ['.txt', '.text']:
            return 'text'
        elif suffix in ['.jpg', '.jpeg']:
            return 'jpeg'
        elif suffix in ['.png']:
            return 'png'
        elif suffix in ['.mp4', '.avi', '.mov', '.mkv']:
            return 'video'
        else:
            return 'binary'
    
    def __iter__(self) -> Iterator[bytes]:
        """迭代器接口"""
        with open(self.file_path, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk
    
    def get_file_info(self) -> dict:
        """获取文件信息"""
        return {
            'path': str(self.file_path),
            'name': self.file_path.name,
            'size': self.file_size,
            'type': self.file_type,
            'chunk_size': self.chunk_size,
            'total_chunks': (self.file_size + self.chunk_size - 1) // self.chunk_size
        }


class TextFileIterator(FileIterator):
    """文本文件迭代器"""
    
    def __init__(self, file_path: str, chunk_size: int = 4200, encoding: str = 'utf-8'):
        super().__init__(file_path, chunk_size)
        self.encoding = encoding
        if self.file_type != 'text':
            raise ValueError(f"不是文本文件: {file_path}")
    
    def __iter__(self) -> Iterator[bytes]:
        """文本文件迭代"""
        with open(self.file_path, 'r', encoding=self.encoding) as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk.encode(self.encoding)


class ImageFileIterator(FileIterator):
    """图像文件迭代器"""
    
    def __init__(self, file_path: str, chunk_size: int = 4200):
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.file_size = self.file_path.stat().st_size
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        self.file_type = self._detect_image_type()
    
    def _detect_image_type(self) -> str:
        """检测图像类型"""
        with open(self.file_path, 'rb') as f:
            header = f.read(10)
            
            if header.startswith(b'\xff\xd8'):
                return 'jpeg'
            elif header.startswith(b'\x89\x50\x4e\x47'):
                return 'png'
            else:
                return 'unknown'
    
    def __iter__(self) -> Iterator[bytes]:
        """图像文件迭代"""
        return FileIterator(str(self.file_path), self.chunk_size).__iter__()
    
    def get_image_info(self) -> Optional[dict]:
        """获取图像信息"""
        try:
            if self.file_type == 'jpeg':
                return self._get_jpeg_info()
            elif self.file_type == 'png':
                return self._get_png_info()
            else:
                return None
        except:
            return None
    
    def _get_jpeg_info(self) -> dict:
        """获取JPEG图像信息"""
        with open(self.file_path, 'rb') as f:
            # 跳过SOI标记
            f.read(2)
            
            while True:
                marker = f.read(2)
                if len(marker) < 2:
                    break
                
                if marker == b'\xff\xc0':  # SOF0标记
                    length = struct.unpack('>H', f.read(2))[0]
                    data = f.read(length - 2)
                    
                    height = struct.unpack('>H', data[1:3])[0]
                    width = struct.unpack('>H', data[3:5])[0]
                    
                    return {
                        'type': 'jpeg',
                        'width': width,
                        'height': height,
                        'channels': 3  # 假设是RGB
                    }
                else:
                    # 跳过其他标记
                    length = struct.unpack('>H', f.read(2))[0]
                    f.seek(length - 2, 1)
        
        return None
    
    def _get_png_info(self) -> dict:
        """获取PNG图像信息"""
        with open(self.file_path, 'rb') as f:
            # 跳过PNG签名
            f.read(8)
            
            # 读取IHDR块
            length = struct.unpack('>I', f.read(4))[0]
            chunk_type = f.read(4)
            
            if chunk_type == b'IHDR':
                width = struct.unpack('>I', f.read(4))[0]
                height = struct.unpack('>I', f.read(4))[0]
                bit_depth = struct.unpack('B', f.read(1))[0]
                color_type = struct.unpack('B', f.read(1))[0]
                
                # 计算通道数
                channels = 1
                if color_type & 2:  # RGB
                    channels = 3
                if color_type & 4:  # 有alpha通道
                    channels += 1
                
                return {
                    'type': 'png',
                    'width': width,
                    'height': height,
                    'bit_depth': bit_depth,
                    'color_type': color_type,
                    'channels': channels
                }
        
        return None


class VideoFileIterator(FileIterator):
    """视频文件迭代器"""
    
    def __init__(self, file_path: str, chunk_size: int = 8400):
        super().__init__(file_path, chunk_size)
        if self.file_type not in ['video']:
            raise ValueError(f"不是视频文件: {file_path}")


class FileProcessor:
    """文件处理器主类"""
    
    def __init__(self, chunk_size: int = 4200):
        """
        初始化文件处理器
        Args:
            chunk_size: 默认块大小
        """
        self.chunk_size = chunk_size
    
    def create_iterator(self, file_path: str, chunk_size: Optional[int] = None) -> FileIterator:
        """
        创建文件迭代器
        Args:
            file_path: 文件路径
            chunk_size: 块大小（可选）
        Returns:
            文件迭代器实例
        """
        chunk_size = chunk_size or self.chunk_size
        
        # 根据文件类型选择合适的迭代器
        suffix = Path(file_path).suffix.lower()
        
        # 为避免编码错误（如UTF-8解码失败），统一按二进制方式读取发送
        if suffix in ['.jpg', '.jpeg', '.png']:
            return ImageFileIterator(file_path, chunk_size)
        elif suffix in ['.mp4', '.avi', '.mov', '.mkv']:
            return VideoFileIterator(file_path, chunk_size)
        else:
            # 包括 .txt 在内的其他类型，统一使用二进制迭代器
            return FileIterator(file_path, chunk_size)
    
    def get_supported_file_types(self) -> list:
        """获取支持的文件类型"""
        return [
            '.txt', '.text',      # 文本文件
            '.jpg', '.jpeg',      # JPEG图像
            '.png',               # PNG图像
            '.mp4', '.avi',       # 视频文件
            '.mov', '.mkv'
        ]
    
    def validate_file(self, file_path: str) -> tuple:
        """
        验证文件
        Args:
            file_path: 文件路径
        Returns:
            (是否有效, 错误信息)
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return False, "文件不存在"
            
            if not path.is_file():
                return False, "路径不是文件"
            
            if path.stat().st_size == 0:
                return False, "文件为空"
            
            suffix = path.suffix.lower()
            supported_types = self.get_supported_file_types()
            
            if suffix not in supported_types:
                return False, f"不支持的文件类型: {suffix}"
            
            return True, ""
            
        except Exception as e:
            return False, f"验证文件时出错: {str(e)}"
    
    def estimate_transmission_time(self, file_path: str, bitrate_bps: int) -> float:
        """
        估算传输时间
        Args:
            file_path: 文件路径
            bitrate_bps: 传输速率（比特每秒）
        Returns:
            估算的传输时间（秒）
        """
        try:
            file_size = Path(file_path).stat().st_size
            return file_size * 8 / bitrate_bps  # 转换为比特
        except:
            return 0.0
    
    def get_file_metadata(self, file_path: str) -> dict:
        """
        获取文件元数据
        Args:
            file_path: 文件路径
        Returns:
            文件元数据
        """
        try:
            path = Path(file_path)
            stat = path.stat()
            
            metadata = {
                'path': str(path),
                'name': path.name,
                'size': stat.st_size,
                'modified_time': stat.st_mtime,
                'created_time': stat.st_ctime,
                'extension': path.suffix.lower(),
                'is_readable': os.access(path, os.R_OK)
            }
            
            # 如果是图像文件，添加图像信息
            if metadata['extension'] in ['.jpg', '.jpeg', '.png']:
                try:
                    img_iterator = ImageFileIterator(file_path)
                    img_info = img_iterator.get_image_info()
                    if img_info:
                        metadata['image_info'] = img_info
                except:
                    pass
            
            return metadata
            
        except Exception as e:
            return {'error': str(e)}


def test_file_processing():
    """测试文件处理功能"""
    processor = FileProcessor(chunk_size=4200)
    
    print("=== 文件处理器测试 ===")
    print(f"支持的文件类型: {processor.get_supported_file_types()}")
    
    # 创建测试文件
    test_files = [
        "test.txt",
        "test.jpg",
        "test.png"
    ]
    
    for filename in test_files:
        if os.path.exists(filename):
            print(f"\n测试文件: {filename}")
            
            # 验证文件
            is_valid, message = processor.validate_file(filename)
            print(f"  文件验证: {'通过' if is_valid else '失败'} - {message}")
            
            if is_valid:
                # 获取文件信息
                metadata = processor.get_file_metadata(filename)
                print(f"  文件大小: {metadata['size']} 字节")
                print(f"  文件类型: {metadata['extension']}")
                
                # 创建迭代器
                iterator = processor.create_iterator(filename)
                print(f"  迭代器信息: {iterator.get_file_info()}")
                
                # 估算传输时间
                estimated_time = processor.estimate_transmission_time(filename, 1000000)  # 1 Mbps
                print(f"  估算传输时间: {estimated_time:.2f} 秒 (1 Mbps)")
                
                # 测试迭代
                chunk_count = 0
                total_size = 0
                for chunk in iterator:
                    chunk_count += 1
                    total_size += len(chunk)
                    if chunk_count <= 3:  # 只显示前几个块
                        print(f"    块 {chunk_count}: {len(chunk)} 字节")
                
                print(f"  总块数: {chunk_count}, 总大小: {total_size} 字节")


if __name__ == "__main__":
    test_file_processing()