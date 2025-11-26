"""
CRC校验模块
实现循环冗余校验功能，支持多种CRC标准
"""

import numpy as np
from typing import Union, List


class CRCCalculator:
    """CRC计算器类"""
    
    def __init__(self, polynomial: str = "CRC-32"):
        """
        初始化CRC计算器
        Args:
            polynomial: CRC多项式名称或自定义多项式
        """
        self.polynomial_name = polynomial
        self._setup_polynomial()
    
    def _setup_polynomial(self):
        """设置CRC多项式参数"""
        # 常见CRC标准的多项式
        crc_standards = {
            "CRC-8": {
                "poly": 0x07,      # x^8 + x^2 + x + 1
                "width": 8,
                "init": 0x00,
                "refin": False,
                "refout": False,
                "xorout": 0x00
            },
            "CRC-16": {
                "poly": 0x8005,    # x^16 + x^15 + x^2 + 1
                "width": 16,
                "init": 0x0000,
                "refin": True,
                "refout": True,
                "xorout": 0x0000
            },
            "CRC-16-CCITT": {
                "poly": 0x1021,    # x^16 + x^12 + x^5 + 1
                "width": 16,
                "init": 0xFFFF,
                "refin": False,
                "refout": False,
                "xorout": 0x0000
            },
            "CRC-32": {
                "poly": 0x04C11DB7,  # x^32 + x^26 + x^23 + x^22 + x^16 + x^12 + x^11 + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1
                "width": 32,
                "init": 0xFFFFFFFF,
                "refin": True,
                "refout": True,
                "xorout": 0xFFFFFFFF
            }
        }
        
        if self.polynomial_name in crc_standards:
            params = crc_standards[self.polynomial_name]
            self.width = params["width"]
            self.poly = params["poly"]
            self.init = params["init"]
            self.refin = params["refin"]
            self.refout = params["refout"]
            self.xorout = params["xorout"]
        else:
            # 自定义多项式
            try:
                # 解析自定义多项式字符串，如 "0x04C11DB7"
                if self.polynomial_name.startswith("0x"):
                    self.poly = int(self.polynomial_name, 16)
                else:
                    self.poly = int(self.polynomial_name)
                
                # 自动检测多项式宽度
                self.width = self.poly.bit_length() - 1
                self.init = 0
                self.refin = False
                self.refout = False
                self.xorout = 0
            except:
                raise ValueError(f"不支持的多项式: {self.polynomial_name}")
        
        # 预计算查找表以提高性能
        self._precompute_table()
        self.crc = self.init
    
    def _precompute_table(self):
        """预计算CRC查找表"""
        self.table = []
        
        for i in range(256):
            crc = i
            if self.refin:
                crc = self._reflect(crc, 8)
            
            crc = crc << (self.width - 8)
            
            for _ in range(8):
                if crc & (1 << (self.width - 1)):
                    crc = (crc << 1) ^ self.poly
                else:
                    crc = crc << 1
                crc = crc & ((1 << self.width) - 1)
            
            if self.refin:
                crc = self._reflect(crc, self.width)
            
            self.table.append(crc)
    
    def _reflect(self, data: int, width: int) -> int:
        """反射比特位"""
        reflected = 0
        for i in range(width):
            if data & (1 << i):
                reflected |= (1 << (width - 1 - i))
        return reflected
    
    def reset(self):
        """重置CRC计算状态"""
        self.crc = self.init

    def update(self, data: bytes):
        """
        更新CRC值
        Args:
            data: 输入数据块
        """
        for byte in data:
            if self.refin:
                self.crc = self.table[(self.crc ^ byte) & 0xFF] ^ (self.crc >> 8)
            else:
                self.crc = self.table[((self.crc >> (self.width - 8)) ^ byte) & 0xFF] ^ (self.crc << 8)
                self.crc = self.crc & ((1 << self.width) - 1)

    def finalize(self) -> int:
        """
        完成CRC计算并返回最终值
        Returns:
            最终CRC校验值
        """
        final_crc = self.crc
        if self.refout:
            final_crc = self._reflect(final_crc, self.width)
        
        final_crc = final_crc ^ self.xorout
        return final_crc

    def calculate(self, data: bytes) -> int:
        """
        计算数据的CRC值
        Args:
            data: 输入数据
        Returns:
            CRC校验值
        """
        self.reset()
        self.update(data)
        return self.finalize()
    
    def verify(self, data: bytes, expected_crc: int) -> bool:
        """
        验证数据的CRC值
        Args:
            data: 输入数据
            expected_crc: 期望的CRC值
        Returns:
            CRC验证结果
        """
        calculated_crc = self.calculate(data)
        return calculated_crc == expected_crc
    
    def append_crc(self, data: bytes) -> bytes:
        """
        将CRC值附加到数据末尾
        Args:
            data: 输入数据
        Returns:
            附加CRC后的数据
        """
        crc = self.calculate(data)
        
        # 将CRC值转换为字节
        crc_bytes = crc.to_bytes((self.width + 7) // 8, byteorder='big')
        
        return data + crc_bytes
    
    def extract_and_verify(self, data_with_crc: bytes) -> tuple:
        """
        从数据中分离CRC并验证
        Args:
            data_with_crc: 包含CRC的数据
        Returns:
            (原始数据, CRC验证结果)
        """
        crc_bytes_length = (self.width + 7) // 8
        
        if len(data_with_crc) < crc_bytes_length:
            return data_with_crc, False
        
        data = data_with_crc[:-crc_bytes_length]
        crc_bytes = data_with_crc[-crc_bytes_length:]
        
        # 将CRC字节转换回整数
        received_crc = int.from_bytes(crc_bytes, byteorder='big')
        
        # 验证CRC
        is_valid = self.verify(data, received_crc)
        
        return data, is_valid


class CRCChecker:
    """CRC检查器主类"""
    
    def __init__(self, polynomial: str = "CRC-32"):
        """
        初始化CRC检查器
        Args:
            polynomial: CRC多项式
        """
        self.calculator = CRCCalculator(polynomial)
    
    def generate_crc(self, data: bytes) -> int:
        """
        生成数据的CRC校验码
        Args:
            data: 输入数据
        Returns:
            CRC校验码
        """
        return self.calculator.calculate(data)
    
    def check_integrity(self, data: bytes, crc: int) -> bool:
        """
        检查数据完整性
        Args:
            data: 输入数据
            crc: CRC校验码
        Returns:
            数据完整性检查结果
        """
        return self.calculator.verify(data, crc)
    
    def add_crc_to_data(self, data: bytes) -> bytes:
        """
        为数据添加CRC校验码
        Args:
            data: 输入数据
        Returns:
            包含CRC校验码的数据
        """
        return self.calculator.append_crc(data)
    
    def verify_and_extract_data(self, data_with_crc: bytes) -> tuple:
        """
        验证数据CRC并提取原始数据
        Args:
            data_with_crc: 包含CRC校验码的数据
        Returns:
            (原始数据, 验证结果)
        """
        return self.calculator.extract_and_verify(data_with_crc)


class RetransmissionManager:
    """重传管理器"""
    
    def __init__(self, max_retransmissions: int = 5):
        """
        初始化重传管理器
        Args:
            max_retransmissions: 最大重传次数
        """
        self.max_retransmissions = max_retransmissions
        self.retransmission_count = 0
        self.successful_transmissions = 0
        self.failed_transmissions = 0
    
    def can_retransmit(self) -> bool:
        """
        检查是否还可以重传
        Returns:
            是否可以重传
        """
        return self.retransmission_count < self.max_retransmissions
    
    def record_transmission_attempt(self, success: bool):
        """
        记录传输尝试结果
        Args:
            success: 传输是否成功
        """
        if success:
            self.successful_transmissions += 1
            self.retransmission_count = 0  # 成功后重置重传计数
        else:
            self.retransmission_count += 1
            if self.retransmission_count >= self.max_retransmissions:
                self.failed_transmissions += 1
    
    def get_statistics(self) -> dict:
        """
        获取传输统计信息
        Returns:
            传输统计信息
        """
        total_attempts = self.successful_transmissions + self.failed_transmissions
        success_rate = (self.successful_transmissions / total_attempts) if total_attempts > 0 else 0
        
        return {
            'max_retransmissions': self.max_retransmissions,
            'current_retransmission_count': self.retransmission_count,
            'successful_transmissions': self.successful_transmissions,
            'failed_transmissions': self.failed_transmissions,
            'total_attempts': total_attempts,
            'success_rate': success_rate
        }
    
    def reset(self):
        """重置所有统计信息"""
        self.retransmission_count = 0
        self.successful_transmissions = 0
        self.failed_transmissions = 0


class ErrorDetectionSystem:
    """错误检测系统主类"""
    """
    集成CRC校验和重传管理的完整错误检测系统
    """
    
    def __init__(self, polynomial: str = "CRC-32", max_retransmissions: int = 5):
        """
        初始化错误检测系统
        Args:
            polynomial: CRC多项式
            max_retransmissions: 最大重传次数
        """
        self.crc_checker = CRCChecker(polynomial)
        self.retransmission_manager = RetransmissionManager(max_retransmissions)
    
    def transmit_with_error_detection(self, data: bytes) -> tuple:
        """
        使用错误检测机制传输数据
        Args:
            data: 要传输的数据
        Returns:
            (传输后的数据, 传输结果)
        """
        # 添加CRC校验码
        data_with_crc = self.crc_checker.calculator.append_crc(data)
        
        return data_with_crc, True
    
    def receive_with_error_detection(self, received_data: bytes) -> tuple:
        """
        使用错误检测机制接收数据
        Args:
            received_data: 接收到的数据
        Returns:
            (原始数据, 接收结果)
        """
        # 验证CRC并提取原始数据
        original_data, is_valid = self.crc_checker.verify_and_extract_data(received_data)

        # 记录传输结果
        self.retransmission_manager.record_transmission_attempt(is_valid)

        return original_data, is_valid
    
    def request_retransmission(self) -> bool:
        """
        请求重传
        Returns:
            是否可以重传
        """
        return self.retransmission_manager.can_retransmit()
    
    def get_system_statistics(self) -> dict:
        """
        获取系统统计信息
        Returns:
            系统统计信息
        """
        crc_stats = {
            'polynomial': self.crc_checker.calculator.polynomial_name,
            'width': self.crc_checker.calculator.width
        }
        
        retrans_stats = self.retransmission_manager.get_statistics()
        
        return {
            'crc_info': crc_stats,
            'retransmission_info': retrans_stats
        }
    
    def reset_system(self):
        """重置系统状态"""
        self.retransmission_manager.reset()


def test_crc_functionality():
    """测试CRC功能"""
    # 测试数据
    test_data = b"Hello, this is a test message for CRC checking!"
    
    # 测试不同CRC标准
    crc_standards = ["CRC-8", "CRC-16", "CRC-16-CCITT", "CRC-32"]
    
    print("=== CRC功能测试 ===")
    for standard in crc_standards:
        print(f"\n测试 {standard}:")
        
        crc_calculator = CRCCalculator(standard)
        crc_checker = CRCChecker(standard)
        
        # 计算CRC
        crc_value = crc_calculator.calculate(test_data)
        print(f"  CRC值: 0x{crc_value:08X}")
        
        # 验证正确性
        is_valid = crc_calculator.verify(test_data, crc_value)
        print(f"  验证结果: {'通过' if is_valid else '失败'}")
        
        # 测试带CRC的数据
        data_with_crc = crc_calculator.append_crc(test_data)
        extracted_data, crc_valid = crc_calculator.extract_and_verify(data_with_crc)
        print(f"  附加CRC验证: {'通过' if crc_valid else '失败'}")
        print(f"  数据长度: 原始={len(test_data)}, 带CRC={len(data_with_crc)}")
    
    # 测试错误检测能力
    print(f"\n=== 错误检测测试 ===")
    crc_system = ErrorDetectionSystem("CRC-32")
    
    # 传输正确数据
    transmitted_data, _ = crc_system.transmit_with_error_detection(test_data)
    received_data, is_valid = crc_system.receive_with_error_detection(transmitted_data)
    print(f"正确数据传输: {'成功' if is_valid else '失败'}")
    
    # 模拟传输错误
    corrupted_data = bytearray(transmitted_data)
    if len(corrupted_data) > 4:
        corrupted_data[-5] ^= 0xFF  # 翻转一些比特
    
    received_data, is_valid = crc_system.receive_with_error_detection(bytes(corrupted_data))
    print(f"错误数据传输: {'检测到错误' if not is_valid else '未检测到错误'}")


if __name__ == "__main__":
    test_crc_functionality()