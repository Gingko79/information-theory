import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os

# 添加父目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 现在可以导入backend中的模块
from backend.channel_coding import LinearBlockCode, ConvolutionalCode


def compare_coding_performance():
    """
    不同误码率下两种编码的性能比较
    Performance comparison of two coding schemes under different bit error rates
    """
    # 初始化编码器
    linear_code = LinearBlockCode(7, 4)
    convolutional_code = ConvolutionalCode(7, 4, 3)

    # 生成测试数据
    data_length = 4200  # 测试数据长度（减少以加快计算）
    test_bits = np.random.randint(0, 1, data_length)

    # 编码
    linear_encoded = linear_code.encode(test_bits)
    conv_encoded = convolutional_code.encode(test_bits)

    # 测试不同的信道误码率 (0到0.5，间隔0.005)
    ber_values = np.arange(0, 0.505, 0.005)  # 101个点
    linear_bers = []  # 线性分组码解码后误码率
    conv_bers = []  # 卷积码解码后误码率

    print("不同误码率下两种编码的性能比较")
    print("Performance comparison of two coding schemes under different bit error rates")
    print("=" * 80)
    print(f"{'信道误码率':<10} {'线性分组码误码率':<15} {'卷积码误码率':<15}")
    print(f"{'Channel BER':<10} {'Linear Code BER':<15} {'Conv Code BER':<15}")
    print("-" * 80)

    # 只打印部分结果，避免输出过多
    print_interval = 20  # 每20个点打印一次

    for i, channel_ber in enumerate(ber_values):
        # 复制编码后的数据
        linear_corrupted = linear_encoded.copy()
        conv_corrupted = conv_encoded.copy()

        # 为线性分组码引入错误
        for j in range(len(linear_corrupted)):
            if random.random() < channel_ber:
                linear_corrupted[j] = 1 - linear_corrupted[j]

        # 为卷积码引入错误
        for j in range(len(conv_corrupted)):
            if random.random() < channel_ber:
                conv_corrupted[j] = 1 - conv_corrupted[j]

        # 解码
        linear_decoded = linear_code.decode(linear_corrupted)
        conv_decoded = convolutional_code.decode(conv_corrupted)

        # 计算解码后误码率
        linear_ber_after = np.sum(linear_decoded[:len(test_bits)] != test_bits) / len(test_bits)
        conv_ber_after = np.sum(conv_decoded[:len(test_bits)] != test_bits) / len(test_bits)

        linear_bers.append(linear_ber_after)
        conv_bers.append(conv_ber_after)

        # 只打印部分结果
        if i % print_interval == 0 or channel_ber in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            print(f"{channel_ber:<10.3f} {linear_ber_after:<15.2e} {conv_ber_after:<15.2e}")

    # 绘制性能比较图
    plt.figure(figsize=(12, 8))

    # 主图 - 线性坐标
    plt.subplot(2, 1, 1)
    plt.plot(ber_values, linear_bers, 'o-', linewidth=2, markersize=3, label='Linear Block Code (7,4)')
    plt.plot(ber_values, conv_bers, 's-', linewidth=2, markersize=3, label='Convolutional Code (7,4,3)')
    plt.plot(ber_values, ber_values, '--', linewidth=1, label='Uncoded', color='gray')

    plt.xlabel('Channel Bit Error Rate')
    plt.ylabel('Decoded Bit Error Rate')
    plt.title('Performance Comparison of Two Coding Schemes\n(7,4) Linear Block Code vs (7,4,3) Convolutional Code')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 副图 - 对数坐标
    plt.subplot(2, 1, 2)
    plt.semilogy(ber_values, linear_bers, 'o-', linewidth=2, markersize=3, label='Linear Block Code (7,4)')
    plt.semilogy(ber_values, conv_bers, 's-', linewidth=2, markersize=3, label='Convolutional Code (7,4,3)')
    plt.semilogy(ber_values, ber_values, '--', linewidth=1, label='Uncoded', color='gray')

    plt.xlabel('Channel Bit Error Rate')
    plt.ylabel('Decoded Bit Error Rate (log scale)')
    plt.title('Performance Comparison (Logarithmic Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('coding_performance_comparison.png', dpi=300)
    plt.show()

    # 计算关键点的编码增益
    print("\n关键点编码增益分析 (Key Points Coding Gain Analysis):")
    print("-" * 60)
    key_bers = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    for key_ber in key_bers:
        idx = int(key_ber / 0.005)
        if idx < len(linear_bers) and idx < len(conv_bers):
            linear_ber = linear_bers[idx]
            conv_ber = conv_bers[idx]

            if linear_ber > 0 and conv_ber > 0:
                gain_linear = 10 * np.log10(key_ber / linear_ber) if linear_ber > 0 else float('inf')
                gain_conv = 10 * np.log10(key_ber / conv_ber) if conv_ber > 0 else float('inf')
                print(
                    f"信道误码率 {key_ber:.2f}: 线性分组码增益 = {gain_linear:.2f} dB, 卷积码增益 = {gain_conv:.2f} dB")

    return ber_values, linear_bers, conv_bers


def analyze_coding_threshold():
    """
    分析两种编码的崩溃点（误码率超过0.45的点）
    Analyze the breakdown point of two coding schemes
    """
    # 初始化编码器
    linear_code = LinearBlockCode(7, 4)
    convolutional_code = ConvolutionalCode(7, 4, 3)

    # 生成测试数据
    test_bits = np.random.randint(0, 2, 4200)

    # 编码
    linear_encoded = linear_code.encode(test_bits)
    conv_encoded = convolutional_code.encode(test_bits)

    print("\n编码崩溃点分析 (Coding Breakdown Point Analysis):")
    print("=" * 60)

    # 测试高误码率情况
    high_bers = [0.4, 0.45, 0.5]
    for ber in high_bers:
        # 线性分组码
        linear_corrupted = linear_encoded.copy()
        for i in range(len(linear_corrupted)):
            if random.random() < ber:
                linear_corrupted[i] = 1 - linear_corrupted[i]

        linear_decoded = linear_code.decode(linear_corrupted)
        linear_errors = np.sum(linear_decoded[:len(test_bits)] != test_bits)

        # 卷积码
        conv_corrupted = conv_encoded.copy()
        for i in range(len(conv_corrupted)):
            if random.random() < ber:
                conv_corrupted[i] = 1 - conv_corrupted[i]

        conv_decoded = convolutional_code.decode(conv_corrupted)
        conv_errors = np.sum(conv_decoded[:len(test_bits)] != test_bits)

        print(
            f"误码率 {ber:.2f}: 线性分组码错误数 = {linear_errors}/{len(test_bits)} ({linear_errors / len(test_bits):.2%}), "
            f"卷积码错误数 = {conv_errors}/{len(test_bits)} ({conv_errors / len(test_bits):.2%})")

        # 检查是否达到崩溃点（误码率>45%）
        if linear_errors / len(test_bits) > 0.45:
            print(f"  -> 线性分组码在误码率 {ber:.2f} 时崩溃")
        if conv_errors / len(test_bits) > 0.45:
            print(f"  -> 卷积码在误码率 {ber:.2f} 时崩溃")


if __name__ == "__main__":
    # 运行性能比较
    ber_values, linear_bers, conv_bers = compare_coding_performance()

    # 运行崩溃点分析
    analyze_coding_threshold()

    # 保存结果到文件
    np.savez('coding_performance_results.npz',
             ber_values=ber_values,
             linear_bers=linear_bers,
             conv_bers=conv_bers)

    print("\n结果已保存到 'coding_performance_results.npz'")
    print("图表已保存为 'coding_performance_comparison.png'")