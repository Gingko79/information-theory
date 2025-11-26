"""
BSC信道模拟器
模拟二进制对称信道，支持可配置的误码率
"""

import numpy as np
from typing import Union
import random


class BSCChannel:
    """二进制对称信道模拟器"""
    
    def __init__(self, error_probability: float = 0.1):
        """
        初始化BSC信道
        Args:
            error_probability: 误码率 ε (0 ≤ ε ≤ 0.5)
        """
        if not 0 <= error_probability <= 0.5:
            raise ValueError("误码率必须在 [0, 0.5] 范围内")
        
        self.error_probability = error_probability
        self.total_bits = 0
        self.error_bits = 0
    
    def set_error_probability(self, error_probability: float):
        """
        设置新的误码率
        Args:
            error_probability: 新的误码率
        """
        if not 0 <= error_probability <= 0.5:
            raise ValueError("误码率必须在 [0, 0.5] 范围内")
        
        self.error_probability = error_probability
    
    def transmit(self, input_bits: np.ndarray) -> np.ndarray:
        """
        通过BSC信道传输比特
        Args:
            input_bits: 输入比特数组
        Returns:
            经过信道传输后的比特数组
        """
        if not isinstance(input_bits, np.ndarray):
            input_bits = np.array(input_bits, dtype=int)
        
        # 确保输入是二进制比特
        input_bits = input_bits.astype(int) % 2
        
        # 生成随机错误图样
        random_errors = np.random.random(len(input_bits))
        error_pattern = (random_errors < self.error_probability).astype(int)
        
        # 应用错误图样（比特翻转）
        output_bits = (input_bits + error_pattern) % 2
        
        # 更新统计信息
        self.total_bits += len(input_bits)
        self.error_bits += np.sum(error_pattern)
        
        return output_bits
    
    def transmit_with_statistics(self, input_bits: np.ndarray) -> tuple:
        """
        通过BSC信道传输比特并返回详细统计
        Args:
            input_bits: 输入比特数组
        Returns:
            (output_bits, error_positions, error_count)
        """
        if not isinstance(input_bits, np.ndarray):
            input_bits = np.array(input_bits, dtype=int)
        
        # 确保输入是二进制比特
        input_bits = input_bits.astype(int) % 2
        
        # 生成随机错误图样
        random_errors = np.random.random(len(input_bits))
        error_pattern = (random_errors < self.error_probability).astype(int)
        
        # 应用错误图样（比特翻转）
        output_bits = (input_bits + error_pattern) % 2
        
        # 获取错误位置
        error_positions = np.where(error_pattern == 1)[0]
        error_count = len(error_positions)
        
        # 更新统计信息
        self.total_bits += len(input_bits)
        self.error_bits += error_count
        
        return output_bits, error_positions, error_count
    
    def get_statistics(self) -> dict:
        """
        获取信道统计信息
        Returns:
            包含统计信息的字典
        """
        actual_ber = self.error_bits / self.total_bits if self.total_bits > 0 else 0
        
        return {
            'total_bits_transmitted': self.total_bits,
            'total_errors': self.error_bits,
            'configured_ber': self.error_probability,
            'actual_ber': actual_ber,
            'error_rate_deviation': abs(actual_ber - self.error_probability)
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.total_bits = 0
        self.error_bits = 0
    
    def simulate_burst_errors(self, input_bits: np.ndarray, burst_length: int, 
                            burst_probability: float = None) -> np.ndarray:
        """
        模拟突发错误信道
        Args:
            input_bits: 输入比特数组
            burst_length: 突发错误长度
            burst_probability: 突发错误发生概率（默认为误码率）
        Returns:
            经过突发错误信道后的比特数组
        """
        if burst_probability is None:
            burst_probability = self.error_probability
        
        if not isinstance(input_bits, np.ndarray):
            input_bits = np.array(input_bits, dtype=int)
        
        input_bits = input_bits.astype(int) % 2
        output_bits = input_bits.copy()
        
        n_bits = len(input_bits)
        error_count = 0
        
        i = 0
        while i < n_bits:
            # 决定是否发生突发错误
            if np.random.random() < burst_probability:
                # 在突发长度内连续翻转比特
                end_pos = min(i + burst_length, n_bits)
                output_bits[i:end_pos] = 1 - output_bits[i:end_pos]
                error_count += (end_pos - i)
                i = end_pos
            else:
                i += 1
        
        # 更新统计信息
        self.total_bits += n_bits
        self.error_bits += error_count
        
        return output_bits


class GilbertElliottChannel:
    """Gilbert-Elliott信道模型（突发错误模型）"""
    
    def __init__(self, p_good: float = 0.9, p_bad: float = 0.1, 
                 ber_good: float = 0.001, ber_bad: float = 0.1):
        """
        初始化Gilbert-Elliott信道
        Args:
            p_good: 从好状态转移到坏状态的概率
            p_bad: 从坏状态转移到好状态的概率
            ber_good: 好状态下的误码率
            ber_bad: 坏状态下的误码率
        """
        self.p_good = p_good
        self.p_bad = p_bad
        self.ber_good = ber_good
        self.ber_bad = ber_bad
        
        self.state = 'good'  # 初始状态为好状态
        self.total_bits = 0
        self.error_bits = 0
        self.state_transitions = 0
    
    def transmit(self, input_bits: np.ndarray) -> np.ndarray:
        """
        通过Gilbert-Elliott信道传输比特
        Args:
            input_bits: 输入比特数组
        Returns:
            经过信道传输后的比特数组
        """
        if not isinstance(input_bits, np.ndarray):
            input_bits = np.array(input_bits, dtype=int)
        
        input_bits = input_bits.astype(int) % 2
        output_bits = input_bits.copy()
        
        for i in range(len(input_bits)):
            # 状态转移
            if self.state == 'good':
                if np.random.random() < self.p_good:
                    self.state = 'bad'
                    self.state_transitions += 1
            else:  # bad state
                if np.random.random() < self.p_bad:
                    self.state = 'good'
                    self.state_transitions += 1
            
            # 根据当前状态决定误码率
            current_ber = self.ber_good if self.state == 'good' else self.ber_bad
            
            # 生成错误
            if np.random.random() < current_ber:
                output_bits[i] = 1 - output_bits[i]
                self.error_bits += 1
            
            self.total_bits += 1
        
        return output_bits
    
    def get_statistics(self) -> dict:
        """获取信道统计信息"""
        actual_ber = self.error_bits / self.total_bits if self.total_bits > 0 else 0
        
        return {
            'total_bits_transmitted': self.total_bits,
            'total_errors': self.error_bits,
            'actual_ber': actual_ber,
            'state_transitions': self.state_transitions,
            'final_state': self.state
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.total_bits = 0
        self.error_bits = 0
        self.state_transitions = 0
        self.state = 'good'


def calculate_bit_error_rate(original_bits: np.ndarray, received_bits: np.ndarray) -> float:
    """
    计算比特错误率
    Args:
        original_bits: 原始比特数组
        received_bits: 接收比特数组
    Returns:
        比特错误率
    """
    if len(original_bits) != len(received_bits):
        raise ValueError("原始比特和接收比特长度必须相同")
    
    if len(original_bits) == 0:
        return 0.0
    
    errors = np.sum(original_bits != received_bits)
    return errors / len(original_bits)


def simulate_channel_performance(channel, test_data_size: int = 10000, 
                               num_trials: int = 10) -> dict:
    """
    模拟信道性能
    Args:
        channel: 信道实例
        test_data_size: 每次试验的测试数据大小
        num_trials: 试验次数
    Returns:
        性能统计信息
    """
    results = []
    
    for trial in range(num_trials):
        # 生成随机测试数据
        test_data = np.random.randint(0, 2, test_data_size)
        
        # 通过信道传输
        channel.reset_statistics()
        received_data = channel.transmit(test_data)
        
        # 计算实际的BER
        actual_ber = calculate_bit_error_rate(test_data, received_data)
        
        # 获取信道统计
        stats = channel.get_statistics()
        stats['trial'] = trial + 1
        stats['measured_ber'] = actual_ber
        
        results.append(stats)
    
    return {
        'trials': results,
        'average_ber': np.mean([r['measured_ber'] for r in results]),
        'ber_std': np.std([r['measured_ber'] for r in results]),
        'total_bits': sum([r['total_bits_transmitted'] for r in results]),
        'total_errors': sum([r['total_errors'] for r in results])
    }
    
    # just a test