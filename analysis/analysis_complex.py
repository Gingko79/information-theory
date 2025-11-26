"""
信道编码复杂度分析
分析线性分组码和卷积码的编码/解码复杂度
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import numpy as np
import matplotlib.pyplot as plt

from backend.channel_coding import LinearBlockCode, ConvolutionalCode, bytes_to_bits, bits_to_bytes
import random

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

class ComplexityAnalyzer:
    """编码复杂度分析器"""

    def __init__(self):
        self.linear_code_7_4 = LinearBlockCode(7, 4)
        self.conv_code_7_4_3 = ConvolutionalCode(7, 4, 3)

    def analyze_encoding_complexity(self, data_sizes):
        """分析编码复杂度"""
        results = {
            'linear_7_4': {'time': [], 'operations': []},
            'conv_7_4_3': {'time': [], 'operations': []}
        }

        for size in data_sizes:
            print(f"分析数据大小: {size} 比特")

            # 生成测试数据
            test_bits = np.array([random.randint(0, 1) for _ in range(size)], dtype=int)
            test_data = bits_to_bytes(test_bits)

            # 线性分组码编码复杂度分析
            start_time = time.time()
            encoded_linear = self.linear_code_7_4.encode(test_bits)
            linear_time = time.time() - start_time

            # 线性分组码操作数估计
            linear_ops = self._estimate_linear_encoding_ops(test_bits)

            results['linear_7_4']['time'].append(linear_time)
            results['linear_7_4']['operations'].append(linear_ops)

            # 卷积码编码复杂度分析
            start_time = time.time()
            encoded_conv = self.conv_code_7_4_3.encode(test_bits)
            conv_time = time.time() - start_time

            # 卷积码操作数估计
            conv_ops = self._estimate_conv_encoding_ops(test_bits)

            results['conv_7_4_3']['time'].append(conv_time)
            results['conv_7_4_3']['operations'].append(conv_ops)

            print(f"  线性码 - 时间: {linear_time:.6f}s, 操作数: {linear_ops}")
            print(f"  卷积码 - 时间: {conv_time:.6f}s, 操作数: {conv_ops}")

        return results

    def analyze_decoding_complexity(self, data_sizes, error_rate=0.01):
        """分析解码复杂度"""
        results = {
            'linear_7_4': {'time': [], 'operations': []},
            'conv_7_4_3': {'time': [], 'operations': []}
        }

        for size in data_sizes:
            print(f"分析解码数据大小: {size} 比特")

            # 生成测试数据并编码
            test_bits = np.array([random.randint(0, 1) for _ in range(size)], dtype=int)

            encoded_linear = self.linear_code_7_4.encode(test_bits)
            encoded_conv = self.conv_code_7_4_3.encode(test_bits)

            # 引入错误
            corrupted_linear = self._introduce_errors(encoded_linear, error_rate)
            corrupted_conv = self._introduce_errors(encoded_conv, error_rate)

            # 线性分组码解码复杂度
            start_time = time.time()
            decoded_linear = self.linear_code_7_4.decode(corrupted_linear)
            linear_time = time.time() - start_time

            linear_ops = self._estimate_linear_decoding_ops(corrupted_linear)

            results['linear_7_4']['time'].append(linear_time)
            results['linear_7_4']['operations'].append(linear_ops)

            # 卷积码解码复杂度
            start_time = time.time()
            decoded_conv = self.conv_code_7_4_3.decode(corrupted_conv)
            conv_time = time.time() - start_time

            conv_ops = self._estimate_conv_decoding_ops(corrupted_conv)

            results['conv_7_4_3']['time'].append(conv_time)
            results['conv_7_4_3']['operations'].append(conv_ops)

            print(f"  线性码解码 - 时间: {linear_time:.6f}s, 操作数: {linear_ops}")
            print(f"  卷积码解码 - 时间: {conv_time:.6f}s, 操作数: {conv_ops}")

        return results

    def _estimate_linear_encoding_ops(self, data_bits):
        """估计线性分组码编码操作数"""
        n, k = 7, 4
        num_blocks = len(data_bits) // k + (1 if len(data_bits) % k != 0 else 0)

        # 每个块的操作: k × n 次模2乘法和加法
        ops_per_block = k * n  # 矩阵乘法

        return num_blocks * ops_per_block

    def _estimate_conv_encoding_ops(self, data_bits):
        """估计卷积码编码操作数"""
        n, k, m = 7, 4, 3
        num_blocks = len(data_bits) // k + (1 if len(data_bits) % k != 0 else 0)

        # (7,4,3)卷积码: 每个4位输入产生7位输出，每个输出位需要多个异或操作
        # 根据编码器实现，每个输出位平均需要2-3次异或操作
        ops_per_block = 7 * 3  # 7个输出位，每个平均3次异或

        return num_blocks * ops_per_block

    def _estimate_linear_decoding_ops(self, received_bits):
        """估计线性分组码解码操作数"""
        n, k = 7, 4
        num_blocks = len(received_bits) // n

        # 每个块的操作:
        # 1. 伴随式计算: m × n 次操作 (m=3)
        # 2. 查表纠正: O(1)
        # 3. 提取信息位: k 次操作
        ops_per_block = 3 * n + k + 1

        return num_blocks * ops_per_block

    def _estimate_conv_decoding_ops(self, received_bits):
        """估计卷积码解码操作数"""
        n, k = 7, 4
        num_blocks = len(received_bits) // n

        # 简化的最小距离解码:
        # 对每个接收块，尝试所有2^k=16种可能输入
        # 对每种可能，计算7位输出并比较汉明距离
        ops_per_block = 16 * (7 + 7)  # 16种可能 × (编码7位 + 比较7位)

        return num_blocks * ops_per_block

    def _introduce_errors(self, bits, error_rate):
        """引入随机错误"""
        corrupted = bits.copy()
        num_errors = int(len(bits) * error_rate)

        if num_errors > 0:
            error_positions = random.sample(range(len(bits)), num_errors)
            for pos in error_positions:
                corrupted[pos] = 1 - corrupted[pos]

        return corrupted

    def plot_complexity_analysis(self, data_sizes, encoding_results, decoding_results):
        """绘制复杂度分析结果"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 编码时间对比
        ax1.plot(data_sizes, encoding_results['linear_7_4']['time'],
                 'bo-', label='线性分组码(7,4)', linewidth=2)
        ax1.plot(data_sizes, encoding_results['conv_7_4_3']['time'],
                 'ro-', label='卷积码(7,4,3)', linewidth=2)
        ax1.set_xlabel('数据大小 (比特)')
        ax1.set_ylabel('编码时间 (秒)')
        ax1.set_title('编码时间复杂度对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 解码时间对比
        ax2.plot(data_sizes, decoding_results['linear_7_4']['time'],
                 'bo-', label='线性分组码(7,4)', linewidth=2)
        ax2.plot(data_sizes, decoding_results['conv_7_4_3']['time'],
                 'ro-', label='卷积码(7,4,3)', linewidth=2)
        ax2.set_xlabel('数据大小 (比特)')
        ax2.set_ylabel('解码时间 (秒)')
        ax2.set_title('解码时间复杂度对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 编码操作数对比
        ax3.plot(data_sizes, encoding_results['linear_7_4']['operations'],
                 'bo-', label='线性分组码(7,4)', linewidth=2)
        ax3.plot(data_sizes, encoding_results['conv_7_4_3']['operations'],
                 'ro-', label='卷积码(7,4,3)', linewidth=2)
        ax3.set_xlabel('数据大小 (比特)')
        ax3.set_ylabel('估计操作数')
        ax3.set_title('编码操作数复杂度对比')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 解码操作数对比
        ax4.plot(data_sizes, decoding_results['linear_7_4']['operations'],
                 'bo-', label='线性分组码(7,4)', linewidth=2)
        ax4.plot(data_sizes, decoding_results['conv_7_4_3']['operations'],
                 'ro-', label='卷积码(7,4,3)', linewidth=2)
        ax4.set_xlabel('数据大小 (比特)')
        ax4.set_ylabel('估计操作数')
        ax4.set_title('解码操作数复杂度对比')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('coding_complexity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def theoretical_analysis():
    """理论复杂度分析"""
    print("=" * 60)
    print("信道编码理论复杂度分析")
    print("=" * 60)

    print("\n1. 线性分组码 (7,4) Hamming码")
    print("   编码复杂度:")
    print("   - 时间复杂度: O(n × k) 每k位输入")
    print("   - 空间复杂度: O(n × k) 存储生成矩阵")
    print("   - 具体: 每个4位块需要7×4=28次模2运算")

    print("\n   解码复杂度:")
    print("   - 时间复杂度: O(m × n) 伴随式计算 + O(1) 查表")
    print("   - 空间复杂度: O(2^m) 存储伴随式表 (m=3, 8个条目)")
    print("   - 具体: 每个7位块需要3×7=21次伴随式计算 + 查表操作")

    print("\n2. 卷积码 (7,4,3)")
    print("   编码复杂度:")
    print("   - 时间复杂度: O(n × 2^m) 每k位输入")
    print("   - 空间复杂度: O(2^m) 存储状态")
    print("   - 具体: 每个4位块需要7个输出位，每个输出位需要多个异或操作")

    print("\n   解码复杂度 (简化的最小距离解码):")
    print("   - 时间复杂度: O(2^k × n) 每n位接收")
    print("   - 空间复杂度: O(1)")
    print("   - 具体: 每个7位接收块需要尝试16种可能输入")

    print("\n3. 复杂度对比总结")
    print("   - 编码: 线性分组码通常更简单，卷积码稍复杂")
    print("   - 解码: 线性分组码有固定低复杂度，卷积码复杂度较高")
    print("   - 纠错能力: 卷积码通常有更好的纠错性能")
    print("   - 适用场景:")
    print("     * 线性分组码: 要求低复杂度、固定延迟的场景")
    print("     * 卷积码: 要求高纠错性能、可接受较高复杂度的场景")


def main():
    """主分析函数"""
    analyzer = ComplexityAnalyzer()

    # 测试数据大小
    data_sizes = [10000, 20000, 50000, 100000, 200000, 500000]

    print("开始编码复杂度分析...")
    encoding_results = analyzer.analyze_encoding_complexity(data_sizes)

    print("\n开始解码复杂度分析...")
    decoding_results = analyzer.analyze_decoding_complexity(data_sizes)

    # 绘制结果
    analyzer.plot_complexity_analysis(data_sizes, encoding_results, decoding_results)

    # 理论分析
    theoretical_analysis()

    # 输出详细分析报告
    print("\n" + "=" * 60)
    print("详细复杂度分析报告")
    print("=" * 60)

    for i, size in enumerate(data_sizes):
        print(f"\n数据大小: {size} 比特")
        print(f"线性分组码(7,4):")
        print(f"  编码时间: {encoding_results['linear_7_4']['time'][i]:.6f}s")
        print(f"  解码时间: {decoding_results['linear_7_4']['time'][i]:.6f}s")
        print(
            f"  总操作数: {encoding_results['linear_7_4']['operations'][i] + decoding_results['linear_7_4']['operations'][i]}")

        print(f"卷积码(7,4,3):")
        print(f"  编码时间: {encoding_results['conv_7_4_3']['time'][i]:.6f}s")
        print(f"  解码时间: {decoding_results['conv_7_4_3']['time'][i]:.6f}s")
        print(
            f"  总操作数: {encoding_results['conv_7_4_3']['operations'][i] + decoding_results['conv_7_4_3']['operations'][i]}")

        # 计算速度比
        encode_speed_ratio = encoding_results['conv_7_4_3']['time'][i] / encoding_results['linear_7_4']['time'][i]
        decode_speed_ratio = decoding_results['conv_7_4_3']['time'][i] / decoding_results['linear_7_4']['time'][i]

        print(f"复杂度比 (卷积码/线性码):")
        print(f"  编码: {encode_speed_ratio:.2f}x")
        print(f"  解码: {decode_speed_ratio:.2f}x")


if __name__ == "__main__":
    main()