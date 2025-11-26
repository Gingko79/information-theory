"""
信道编码模块
支持线性分组码和卷积码编码解码
"""
import random

import numpy as np
import itertools


class LinearBlockCode:
    """线性分组码实现类"""

    def __init__(self, n: int, k: int):
        """
        初始化线性分组码
        Args:
            n: 码字长度
            k: 信息位长度
        """
        self.n = n
        self.k = k
        self._setup_matrices()

    def _setup_matrices(self):
        """设置生成矩阵和监督矩阵"""
        if self.n == 7 and self.k == 4:
            # (7,4) Hamming code - 矩阵定义正确
            self.G = np.array([
                [1, 0, 0, 0, 1, 1, 0],
                [0, 1, 0, 0, 1, 0, 1],
                [0, 0, 1, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 1, 1]
            ], dtype=int)

            self.H = np.array([
                [1, 1, 0, 1, 1, 0, 0],
                [1, 0, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, 0, 0, 1]
            ], dtype=int)

            # 伴随式表需要补充双错误情况
            self.syndrome_table = {
                tuple([0, 0, 0]): np.array([0, 0, 0, 0, 0, 0, 0]),
                tuple([0, 0, 1]): np.array([0, 0, 0, 0, 0, 0, 1]),
                tuple([0, 1, 0]): np.array([0, 0, 0, 0, 0, 1, 0]),
                tuple([0, 1, 1]): np.array([0, 0, 0, 0, 1, 0, 0]),
                tuple([1, 0, 0]): np.array([0, 0, 0, 1, 0, 0, 0]),  # 修正：应该是位置3
                tuple([1, 0, 1]): np.array([0, 0, 1, 0, 0, 0, 0]),  # 修正：应该是位置2
                tuple([1, 1, 0]): np.array([0, 1, 0, 0, 0, 0, 0]),  # 修正：应该是位置1
                tuple([1, 1, 1]): np.array([1, 0, 0, 0, 0, 0, 0])  # 修正：应该是位置0
            }

        elif self.n == 3 and self.k == 2:
            # (3,2) 简单奇偶校验码 - 只能检测错误，不能纠正
            self.G = np.array([
                [1, 0, 1],
                [0, 1, 1]
            ], dtype=int)

            self.H = np.array([[1, 1, 1]], dtype=int)

            # (3,2)码只能检测单比特错误，不能确定位置
            self.syndrome_table = {
                tuple([0]): np.array([0, 0, 0]),
                # 对于伴随式[1]，我们不知道错误位置，只能标记错误
                tuple([1]): np.array([0, 0, 0])  # 无法纠正，只能检测
            }
        else:
            raise ValueError(f"不支持 ({self.n},{self.k}) 线性分组码")

    def encode(self, data_bits: np.ndarray) -> np.ndarray:
        """
        编码数据比特（优化版本）
        Args:
            data_bits: 输入数据比特数组
        Returns:
            编码后的码字
        """
        # 确保数据长度是k的倍数
        if len(data_bits) % self.k != 0:
            padding_length = self.k - (len(data_bits) % self.k)
            data_bits = np.concatenate([data_bits, np.zeros(padding_length, dtype=int)])

        # 将数据分成k位一组（使用view而不是copy）
        num_groups = len(data_bits) // self.k
        data_groups = data_bits.reshape(num_groups, self.k)

        # 使用矩阵乘法批量编码，比循环快得多
        # G的形状是(k, n)，data_groups的形状是(num_groups, k)
        # 结果形状是(num_groups, n)
        encoded_codewords = np.dot(data_groups, self.G) % 2

        return encoded_codewords.flatten()

    def decode(self, received_codewords: np.ndarray) -> np.ndarray:
        """
        解码接收到的码字
        Args:
            received_codewords: 接收到的码字
        Returns:
            解码后的数据比特
        """
        # 将码字分成n位一组
        if len(received_codewords) % self.n != 0:
            raise ValueError("接收到的码字长度不是n的倍数")

        codeword_groups = received_codewords.reshape(-1, self.n)
        decoded_bits = []

        for codeword in codeword_groups:
            # 使用最小距离解码提高纠错能力
            min_distance = float('inf')
            best_codeword = None
            
            # 对于(7,4)码，尝试所有可能的有效码字
            if self.n == 7 and self.k == 4:
                # 生成所有可能的4位信息位
                for info_bits_candidate in itertools.product([0, 1], repeat=self.k):
                    info_bits_candidate = np.array(list(info_bits_candidate), dtype=int)
                    # 编码得到有效码字
                    valid_codeword = np.dot(info_bits_candidate, self.G) % 2
                    # 计算汉明距离
                    distance = np.sum(codeword != valid_codeword)
                    if distance < min_distance:
                        min_distance = distance
                        best_codeword = valid_codeword
                
                # 使用最佳码字提取信息位
                if best_codeword is not None:
                    info_bits = best_codeword[:self.k]
                else:
                    # 回退到伴随式解码
                    syndrome = np.dot(self.H, codeword) % 2
                    error_pattern = self.syndrome_table.get(tuple(syndrome), np.zeros(self.n))
                    corrected_codeword = (codeword + error_pattern) % 2
                    info_bits = corrected_codeword[:self.k]
            
            # 对于(3,2)码，使用最小距离解码
            elif self.n == 3 and self.k == 2:
                # 生成所有可能的2位信息位
                for info_bits_candidate in itertools.product([0, 1], repeat=self.k):
                    info_bits_candidate = np.array(list(info_bits_candidate), dtype=int)
                    # 编码得到有效码字
                    valid_codeword = np.dot(info_bits_candidate, self.G) % 2
                    # 计算汉明距离
                    distance = np.sum(codeword != valid_codeword)
                    if distance < min_distance:
                        min_distance = distance
                        best_codeword = valid_codeword
                
                # 使用最佳码字提取信息位
                if best_codeword is not None:
                    info_bits = best_codeword[:self.k]
                else:
                    # 回退：直接使用前k位（可能包含错误）
                    info_bits = codeword[:self.k]
            
            else:
                # 其他码使用伴随式解码
                syndrome = np.dot(self.H, codeword) % 2
                error_pattern = self.syndrome_table.get(tuple(syndrome), np.zeros(self.n))
                corrected_codeword = (codeword + error_pattern) % 2
                info_bits = corrected_codeword[:self.k]
            
            decoded_bits.extend(info_bits)

        return np.array(decoded_bits, dtype=int)

    def get_original_length(self, decoded_bits: np.ndarray, original_bit_count: int) -> np.ndarray:
        """
        获取原始长度的数据（去除填充位）
        Args:
            decoded_bits: 解码后的比特
            original_bit_count: 原始数据比特数
        Returns:
            截取到原始长度的比特数组
        """
        return decoded_bits[:original_bit_count]


class ConvolutionalCode:
    """卷积码实现类"""

    def __init__(self, n: int, k: int, m: int):
        """
        初始化卷积码
        Args:
            n: 输出比特数
            k: 输入比特数
            m: 记忆长度
        """
        self.n = n
        self.k = k
        self.m = m
        self.constraint_length = m + 1

        # 设置生成多项式
        if self.n == 2 and self.k == 1 and self.m == 2:
            # (2,1,2) 卷积码，约束长度=3
            # 生成多项式: g0 = [1,1,1], g1 = [1,0,1] (八进制表示为7,5)
            self.generators = [
                [1, 1, 1],  # g0
                [1, 0, 1]  # g1
            ]
            self.rate = k / n  # 1/2
            # 状态转移表
            self.trellis = {
                0: {0: (0, [0, 0]), 1: (1, [1, 1])},  # 状态0
                1: {0: (2, [1, 1]), 1: (3, [0, 0])},  # 状态1
                2: {0: (0, [1, 0]), 1: (1, [0, 1])},  # 状态2
                3: {0: (2, [0, 1]), 1: (3, [1, 0])}  # 状态3
            }

        elif self.n == 7 and self.k == 4 and self.m == 3:
            # (7,4,3) "卷积码" - 实际上是分组码
            self.rate = k / n  # 4/7
            # 使用线性编码方案

        else:
            raise ValueError(f"不支持 ({self.n},{self.k},{self.m}) 卷积码")

    def encode(self, data_bits: np.ndarray) -> np.ndarray:
        """编码实现（优化版本）"""
        if len(data_bits) == 0:
            return np.array([], dtype=int)

        if self.n == 2 and self.k == 1 and self.m == 2:
            # (2,1,2) 卷积码编码（优化版本）
            # 重置移位寄存器
            shift_register = np.zeros(self.m, dtype=int)
            data_bits = data_bits.astype(int)
            
            # 预分配输出数组（避免列表扩展）
            num_output_bits = len(data_bits) * self.n + self.m * self.n  # 数据位 + 尾比特
            encoded_bits = np.zeros(num_output_bits, dtype=int)
            output_idx = 0

            for bit in data_bits:
                # 更新移位寄存器
                shift_register = np.roll(shift_register, 1)
                shift_register[0] = bit

                # 计算两个输出比特
                encoded_bits[output_idx] = (bit + shift_register[0] + shift_register[1]) % 2
                encoded_bits[output_idx + 1] = (bit + shift_register[1]) % 2
                output_idx += 2

            # 添加尾比特清空寄存器
            for _ in range(self.m):
                shift_register = np.roll(shift_register, 1)
                shift_register[0] = 0
                encoded_bits[output_idx] = (shift_register[0] + shift_register[1]) % 2
                encoded_bits[output_idx + 1] = shift_register[1] % 2
                output_idx += 2

            return encoded_bits[:output_idx]  # 返回实际使用的部分

        elif self.n == 7 and self.k == 4 and self.m == 3:
            # (7,4,3) 改进的编码方案（优化版本）
            # 确保数据长度是k的倍数
            if len(data_bits) % self.k != 0:
                padding_length = self.k - (len(data_bits) % self.k)
                data_bits = np.concatenate([data_bits, np.zeros(padding_length, dtype=int)])
            
            # 将数据分成k位一组
            num_groups = len(data_bits) // self.k
            data_groups = data_bits.reshape(num_groups, self.k)
            
            # 预分配输出数组
            encoded_bits = np.zeros(num_groups * self.n, dtype=int)
            
            # 批量编码（使用向量化操作）
            for i, group in enumerate(data_groups):
                base_idx = i * self.n
                encoded_bits[base_idx:base_idx + 4] = group  # 信息位
                # 校验位（使用XOR操作）
                encoded_bits[base_idx + 4] = (group[0] ^ group[1] ^ group[2]) % 2
                encoded_bits[base_idx + 5] = (group[0] ^ group[1] ^ group[3]) % 2
                encoded_bits[base_idx + 6] = (group[0] ^ group[2] ^ group[3]) % 2

            return encoded_bits

        return np.array([], dtype=int)

    def decode(self, received_bits: np.ndarray) -> np.ndarray:
        """
        卷积码解码
        Args:
            received_bits: 接收到的比特流
        Returns:
            解码后的数据比特
        """
        if len(received_bits) == 0:
            return np.array([], dtype=int)

        # 确保接收到的比特数是n的倍数
        if len(received_bits) % self.n != 0:
            received_bits = received_bits[:-(len(received_bits) % self.n)]

        decoded_bits = []

        if self.n == 2 and self.k == 1 and self.m == 2:
            # (2,1,2) 卷积码的Viterbi解码算法
            received_groups = received_bits.reshape(-1, self.n)
            
            # 计算原始数据长度（去除尾比特）
            num_groups = len(received_groups)
            original_length = max(0, num_groups - self.m)  # 去除尾比特组
            
            if original_length <= 0:
                return np.array([], dtype=int)
            
            # Viterbi算法参数
            num_states = 2 ** self.m  # 4个状态
            trellis_depth = original_length
            
            # 初始化路径度量和路径历史
            # path_metrics[state] = 当前状态的最小累积度量
            path_metrics = np.full(num_states, float('inf'))
            path_metrics[0] = 0  # 初始状态为0
            
            # paths[state] = 到达该状态的最佳路径（输入比特序列）
            paths = [[] for _ in range(num_states)]
            
            # 状态转移和输出函数（与编码逻辑一致）
            def get_output_and_next_state(current_state, input_bit):
                """根据当前状态和输入比特，计算输出和下一个状态"""
                # 从状态恢复寄存器值（roll之前的状态）
                # 状态编码：state = register[0] * 2^0 + register[1] * 2^1
                # 在编码时，roll之前：register[0]是前一个输入，register[1]是前两个输入
                reg0_before = (current_state >> 0) & 1  # roll之前的register[0]
                
                # 模拟编码过程：roll后新输入进入register[0]
                # roll之后：register[0] = input_bit, register[1] = reg0_before
                new_reg0 = input_bit
                new_reg1 = reg0_before
                
                # 计算输出（与编码逻辑完全一致）
                # 编码时：output0 = (bit + shift_register[0] + shift_register[1]) % 2
                # 其中shift_register[0] = input_bit, shift_register[1] = reg0_before
                output0 = (input_bit + new_reg0 + new_reg1) % 2
                output1 = (input_bit + new_reg1) % 2
                
                # 计算下一个状态（roll之后的状态）
                next_state = (new_reg0 | (new_reg1 << 1)) & (num_states - 1)
                
                return [output0, output1], next_state
            
            # Viterbi算法主循环
            for i in range(trellis_depth):
                group = received_groups[i]
                new_path_metrics = np.full(num_states, float('inf'))
                new_paths = [[] for _ in range(num_states)]
                
                # 对每个当前状态
                for state in range(num_states):
                    if path_metrics[state] == float('inf'):
                        continue
                    
                    # 尝试两种可能的输入比特
                    for input_bit in [0, 1]:
                        expected_output, next_state = get_output_and_next_state(state, input_bit)
                        
                        # 计算分支度量（汉明距离）
                        branch_metric = sum([1 for j in range(2) if group[j] != expected_output[j]])
                        
                        # 累积度量
                        new_metric = path_metrics[state] + branch_metric
                        
                        # 更新路径（如果找到更好的路径）
                        if new_metric < new_path_metrics[next_state]:
                            new_path_metrics[next_state] = new_metric
                            new_paths[next_state] = paths[state] + [input_bit]
                
                path_metrics = new_path_metrics
                paths = new_paths
            
            # 选择最佳路径（最小累积度量）
            best_state = np.argmin(path_metrics)
            decoded_bits = paths[best_state]
            
            # 确保长度正确
            if len(decoded_bits) > original_length:
                decoded_bits = decoded_bits[:original_length]

        elif self.n == 7 and self.k == 4 and self.m == 3:
            # (7,4,3) 卷积码的改进解码 - 使用最小距离解码
            # 确保接收到的比特数是7的倍数
            if len(received_bits) % self.n != 0:
                # 截取到7的倍数（向下取整）
                received_bits = received_bits[:-(len(received_bits) % self.n)]
            
            if len(received_bits) == 0:
                return np.array([], dtype=int)
            
            received_groups = received_bits.reshape(-1, self.n)

            for group in received_groups:
                # 使用改进的解码算法：先尝试直接提取信息位并校验，如果校验失败则使用最小距离解码
                # 提取信息位（前4位）
                info_bits = group[:self.k].copy()
                
                # 计算期望的校验位
                expected_parity0 = (info_bits[0] ^ info_bits[1] ^ info_bits[2]) % 2
                expected_parity1 = (info_bits[0] ^ info_bits[1] ^ info_bits[3]) % 2
                expected_parity2 = (info_bits[0] ^ info_bits[2] ^ info_bits[3]) % 2
                
                # 获取接收到的校验位
                received_parity0 = group[4]
                received_parity1 = group[5]
                received_parity2 = group[6]
                
                # 计算伴随式（校验位错误）
                syndrome = [
                    (expected_parity0 + received_parity0) % 2,
                    (expected_parity1 + received_parity1) % 2,
                    (expected_parity2 + received_parity2) % 2
                ]
                
                # 如果伴随式全为0，说明没有错误或错误在信息位
                if syndrome == [0, 0, 0]:
                    # 检查信息位是否有错误（通过重新编码验证）
                    expected_output = [
                        info_bits[0],
                        info_bits[1],
                        info_bits[2],
                        info_bits[3],
                        expected_parity0,
                        expected_parity1,
                        expected_parity2
                    ]
                    # 如果信息位和校验位都匹配，直接使用
                    if np.array_equal(group, expected_output):
                        decoded_bits.extend(info_bits)
                        continue
                
                # 如果有错误，使用最小距离解码
                min_distance = float('inf')
                best_input = None

                # 尝试所有可能的4位输入（共16种可能）
                for input_candidate in itertools.product([0, 1], repeat=self.k):
                    input_candidate = list(input_candidate)

                    # 计算期望输出（与改进的编码逻辑完全一致）
                    expected_output = [
                        input_candidate[0],  # 信息位0
                        input_candidate[1],  # 信息位1
                        input_candidate[2],  # 信息位2
                        input_candidate[3],  # 信息位3
                        input_candidate[0] ^ input_candidate[1] ^ input_candidate[2],  # 校验位0
                        input_candidate[0] ^ input_candidate[1] ^ input_candidate[3],  # 校验位1
                        input_candidate[0] ^ input_candidate[2] ^ input_candidate[3]  # 校验位2
                    ]

                    # 计算汉明距离
                    distance = sum([1 for i in range(self.n) if group[i] != expected_output[i]])

                    # 选择距离最小的候选
                    if distance < min_distance:
                        min_distance = distance
                        best_input = input_candidate

                # 使用最佳候选
                if best_input is not None:
                    decoded_bits.extend(best_input)
                else:
                    # 如果没有找到候选，使用默认值（这不应该发生）
                    decoded_bits.extend([0, 0, 0, 0])

        return np.array(decoded_bits, dtype=int)

class ChannelEncoder:
    """信道编码器主类"""

    def __init__(self):
        self.encoders = {
            'linear_7_4': LinearBlockCode(7, 4),
            'linear_3_2': LinearBlockCode(3, 2),
            'conv_7_4_3': ConvolutionalCode(7, 4, 3),
            'conv_2_1_2': ConvolutionalCode(2, 1, 2)
        }

    def encode(self, data: bytes, method: str) -> np.ndarray:
        """
        编码数据（优化版本）
        Args:
            data: 输入数据
            method: 编码方法
        Returns:
            编码后的比特数组
        """
        if method not in self.encoders:
            raise ValueError(f"不支持的编码方法: {method}")

        # 使用优化的bytes_to_bits函数
        data_bits = bytes_to_bits(data)

        return self.encoders[method].encode(data_bits)

    def decode(self, encoded_bits: np.ndarray, method: str) -> bytes:
        """
        解码数据（优化版本）
        Args:
            encoded_bits: 编码后的比特数组
            method: 编码方法
        Returns:
            解码后的数据
        """
        if method not in self.encoders:
            raise ValueError(f"不支持的编码方法: {method}")

        decoded_bits = self.encoders[method].decode(encoded_bits)

        # 使用优化的bits_to_bytes函数
        return bits_to_bytes(decoded_bits)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """将比特数组转换为字节（优化版本）"""
    if len(bits) == 0:
        return b''
    
    # 确保是整数类型，并且值只能是0或1
    bits = bits.astype(np.uint8) % 2
    
    # 如果需要填充，使用更高效的方式
    if len(bits) % 8 != 0:
        padding_length = 8 - (len(bits) % 8)
        bits = np.concatenate([bits, np.zeros(padding_length, dtype=np.uint8)])
    
    # 使用numpy的packbits函数，比循环快得多
    # packbits可以直接处理1D数组，按8位一组打包
    byte_array = np.packbits(bits, bitorder='big')
    
    return bytes(byte_array)


def bytes_to_bits(data: bytes) -> np.ndarray:
    """将字节转换为比特数组（优化版本）"""
    if len(data) == 0:
        return np.array([], dtype=np.uint8)
    
    # 使用numpy的unpackbits函数，比列表推导式快得多
    # 先将bytes转换为uint8数组，然后使用unpackbits
    byte_array = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(byte_array)
    return bits.astype(int)



if __name__ == '__main__':
    encoder = ChannelEncoder()

    # 生成4200位随机比特
    print("生成4200位随机测试数据...")
    random_bits = np.array([random.randint(0, 1) for _ in range(4200)], dtype=int)
    print(f"原始数据长度: {len(random_bits)} 比特")

    # 将比特转换为字节（用于编码器输入）
    test_data = bits_to_bytes(random_bits)
    print(f"转换为字节后长度: {len(test_data)} 字节")

    # 测试所有四种编码方法
    methods = ['linear_7_4', 'linear_3_2', 'conv_2_1_2', 'conv_7_4_3']

    for method in methods:
        print(f"\n{'=' * 50}")
        print(f"测试方法: {method}")
        print(f"{'=' * 50}")

        try:
            # 编码
            print("正在进行编码...")
            encoded_bits = encoder.encode(test_data, method)
            print(f"编码后长度: {len(encoded_bits)} 比特")
            print(f"编码率: {len(random_bits) / len(encoded_bits):.3f}")

            # 解码
            print("正在进行解码...")
            decoded_data = encoder.decode(encoded_bits, method)

            # 将解码后的数据转换回比特
            decoded_bits = bytes_to_bits(decoded_data)

            # 截取到原始长度（因为可能有填充）
            decoded_bits = decoded_bits[:len(random_bits)]

            # 检查解码是否正确
            if len(decoded_bits) != len(random_bits):
                print(f"❌ 长度不匹配: 原始{len(random_bits)}比特, 解码后{len(decoded_bits)}比特")
                continue

            # 计算误码率
            error_count = np.sum(decoded_bits != random_bits)
            ber = error_count / len(random_bits)

            if error_count == 0:
                print(f"✅ 测试通过 - 解码完全正确")
                print(f"误码率: {ber:.2e}")
            else:
                print(f"⚠️  解码存在错误")
                print(f"错误比特数: {error_count}")
                print(f"误码率: {ber:.2e}")

            # 显示一些统计信息
            print(f"原始数据前10比特: {random_bits[:10]}")
            print(f"解码数据前10比特: {decoded_bits[:10]}")

        except Exception as e:
            print(f"❌ 测试失败 - 错误信息: {e}")

    print(f"\n{'=' * 50}")
    print("所有测试完成！")
    print(f"{'=' * 50}")

    test_bits = np.array([random.randint(0, 1) for _ in range(1000)], dtype=int)
    test_data = bits_to_bytes(test_bits)

    methods = ['linear_7_4', 'linear_3_2', 'conv_2_1_2', 'conv_7_4_3']

    for method in methods:
        print(f"\n测试方法: {method}")

        try:
            # 编码
            encoded_bits = encoder.encode(test_data, method)

            # 引入随机错误
            error_positions = random.sample(range(len(encoded_bits)), min(10, len(encoded_bits) // 100))
            corrupted_bits = encoded_bits.copy()
            for pos in error_positions:
                corrupted_bits[pos] = 1 - corrupted_bits[pos]  # 翻转比特

            print(f"引入了 {len(error_positions)} 个错误")

            # 解码
            decoded_data = encoder.decode(corrupted_bits, method)
            decoded_bits = bytes_to_bits(decoded_data)[:len(test_bits)]

            # 计算误码率
            error_count = np.sum(decoded_bits != test_bits)
            ber = error_count / len(test_bits)

            if error_count == 0:
                print(f"✅ 纠错成功 - 所有错误被纠正")
            else:
                print(f"⚠️  仍有 {error_count} 个错误未被纠正")
            print(f"最终误码率: {ber:.2e}")

        except Exception as e:
            print(f"❌ 测试失败: {e}")