import numpy as np
from commpy.channelcoding.convcode import Trellis
from commpy.channelcoding.convcode import conv_encode
'''
# 读取二进制文件
def read_binary_file(filename):
    """
    从文件中读取二进制数据
    :param filename: 文件路径
    :return: 二进制数据 (numpy 数组)
    """
    with open(filename, 'r',encoding='utf-8') as f:
        binary_data = f.read()
        print(len(binary_data))
    return np.array([int(bit) for bit in binary_data], dtype=int)

# 保存编码后的数据到文件
def save_encoded_file(filename, encoded_data):
    """
    将编码后的数据保存到文件
    :param filename: 文件路径
    :param encoded_data: 编码后的数据 (numpy 数组)
    """
    with open(filename, 'w') as f:
        for bit in encoded_data:
            f.write(str(bit))

# 卷积码编码
def convolutional_encoding(input_data, trellis):
    """
    使用卷积码对二进制数据进行编码
    :param input_data: 输入的二进制数据 (numpy 数组)
    :param trellis: Trellis 结构
    :return: 编码后的二进制数据
    """
    encoded_data = conv_encode(input_data, trellis)
    print(len(encoded_data))
    return encoded_data

# 输出卷积码信息
def print_convolutional_code_info(trellis):
    """
    输出卷积码的相关信息
    :param trellis: Trellis 结构
    """
    memory = trellis.total_memory  # 从 Trellis 中获取记忆单元数量
    #generator_polynomials = trellis.g
    input_bits = trellis.k
    output_bits = trellis.n
    rate = input_bits / output_bits
    num_states = 2 ** memory

    print("卷积码信息：")
    print(f"  码率 (Rate): {input_bits}/{output_bits} = {rate}")
    #print(f"  生成多项式 (Generator Polynomials): {generator_polynomials}")
    print(f"  约束长度 (Constraint Length): {memory + 1}")
    print(f"  状态数 (Number of States): {num_states}")

# 示例流程
try:
    # 第一步：读取原始二进制文件
    input_file = "C:/Users/zhoupenghua/Desktop/binarymessage.txt"  # 输入文件路径
    binary_data = read_binary_file(input_file)
    print("成功读取二进制文件!")

    # 第二步：定义卷积码 Trellis 结构
    memory = np.array([2])
    g_matrix = np.array([[0b111, 0b101]])
    trellis = Trellis(memory=memory,g_matrix=g_matrix)
    print(g_matrix)
    print_convolutional_code_info(trellis)
    # 第三步：进行卷积码编码
    encoded_data = convolutional_encoding(binary_data, trellis)
    print("卷积码编码完成!")

    # 第四步：保存编码后的数据到新文件
    output_file = "C:/Users/zhoupenghua/Desktop/conveencode.txt"  # 输出文件路径
    save_encoded_file(output_file, encoded_data)
    print(f"编码后的数据已保存到文件 {output_file}")

except Exception as e:
    print(f"发生错误: {e}")
'''
'''
import numpy as np
from commpy.channelcoding.convcode import Trellis
from commpy.channelcoding.convcode import conv_encode

# 读取二进制文件
def read_binary_file(filename):
    """
    从文件中读取二进制数据
    :param filename: 文件路径
    :return: 二进制数据 (numpy 数组)
    """
    with open(filename, 'r', encoding='utf-8') as f:
        binary_data = f.read()
        print(f"读取的比特数: {len(binary_data)}")
    return np.array([int(bit) for bit in binary_data], dtype=int)

# 保存编码后的数据到文件
def save_encoded_file(filename, encoded_data):
    """
    将编码后的数据保存到文件
    :param filename: 文件路径
    :param encoded_data: 编码后的数据 (numpy 数组)
    """
    with open(filename, 'w') as f:
        for bit in encoded_data:
            f.write(str(bit))

# 卷积码编码
def convolutional_encoding(input_data, trellis):
    """
    使用卷积码对二进制数据进行编码
    :param input_data: 输入的二进制数据 (numpy 数组)
    :param trellis: Trellis 结构
    :return: 编码后的二进制数据
    """
    encoded_data = conv_encode(input_data, trellis)
    print(f"编码后的比特数: {len(encoded_data)}")
    return encoded_data

# 输出卷积码信息
def print_convolutional_code_info(trellis):
    """
    输出卷积码的相关信息
    :param trellis: Trellis 结构
    """
    memory = trellis.total_memory  # 从 Trellis 中获取记忆单元数量
    input_bits = trellis.k
    output_bits = trellis.n
    rate = input_bits / output_bits
    num_states = 2 ** memory

    print("卷积码信息：")
    print(f"  码率 (Rate): {input_bits}/{output_bits} = {rate}")
    print(f"  约束长度 (Constraint Length): {memory + 1}")
    print(f"  状态数 (Number of States): {num_states}")

# 示例流程
try:
    # 第一步：读取原始二进制文件
    input_file = "C:/Users/zhoupenghua/Desktop/binarymessage.txt"  # 输入文件路径
    binary_data = read_binary_file(input_file)
    print("成功读取二进制文件!")

    # 第二步：定义卷积码 Trellis 结构
    memory = np.array([3])  # 选择合理的记忆长度
    g_matrix = np.array([[0b111, 0b101, 0b110, 0b100]])  # 生成多项式，适应码率为1/4
    trellis = Trellis(memory=memory, g_matrix=g_matrix)
    print(g_matrix)
    print_convolutional_code_info(trellis)

    # 第三步：进行卷积码编码
    encoded_data = convolutional_encoding(binary_data, trellis)
    print("卷积码编码完成!")

    # 第四步：保存编码后的数据到新文件
    output_file = "C:/Users/zhoupenghua/Desktop/conveencode2.txt"  # 输出文件路径
    save_encoded_file(output_file, encoded_data)
    print(f"编码后的数据已保存到文件 {output_file}")

except Exception as e:
    print(f"发生错误: {e}")'''
import numpy as np
from commpy.channelcoding.convcode import Trellis
from commpy.channelcoding.convcode import conv_encode

def read_binary_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        binary_data = f.read().strip()
    return np.array([int(bit) for bit in binary_data if bit in '01'], dtype=int)

def save_encoded_file(filename, encoded_data):
    with open(filename, 'w') as f:
        for bit in encoded_data:
            f.write(str(bit))

def convolutional_encoding(input_data, trellis):
    encoded_data = conv_encode(input_data, trellis)
    return encoded_data

def print_convolutional_code_info(trellis):
    memory = trellis.total_memory
    input_bits = trellis.k
    output_bits = trellis.n
    rate = input_bits / output_bits
    print("卷积码信息：")
    print(f"  输入比特数 (k): {input_bits}")
    print(f"  输出比特数 (n): {output_bits}")
    print(f"  码率 (Rate): {input_bits}/{output_bits} = {rate:.2f}")
    print(f"  约束长度 (Constraint Length): {memory + 1}")
    print(f"  状态数 (Number of States): {2 ** memory}")

# 示例流程
try:
    input_file = "C:/Users/zhoupenghua/Desktop/binarymessage.txt"
    binary_data = read_binary_file(input_file)

    print("成功读取二进制文件!")

    # 填充输入数据使其长度为48的倍数
    block_size = 48
    if len(binary_data) < block_size:
        raise ValueError("输入数据长度少于48比特，请提供更长的二进制数据。")

    # 确保二进制数据的长度是48的倍数（如果有多余的部分可以截断或填充）
    if len(binary_data) % block_size != 0:
        binary_data = np.pad(binary_data, (0, block_size - len(binary_data) % block_size), 'constant', constant_values=(0,))

    # 修改为3/4码率的生成多项式和记忆长度
    memory = np.array([6])  # 适当的记忆长度
    g_matrix = np.array([[0b111, 0b101]])  # 示例生成多项式

    # 创建 Trellis
    trellis = Trellis(memory=memory, g_matrix=g_matrix)

    # 输出卷积码信息
    print_convolutional_code_info(trellis)

    # 分块处理，将长的二进制数据分成48比特的块
    encoded_result = []

    # 逐块处理输入数据
    for i in range(0, len(binary_data), block_size):
        input_block = binary_data[i:i + block_size]
        # 进行卷积码编码
        encoded_block = convolutional_encoding(input_block, trellis)
        encoded_result.extend(encoded_block)

    print("卷积码编码完成!")

    # 保存编码后的数据到新文件
    output_file = "C:/Users/zhoupenghua/Desktop/conveencode3.txt"
    save_encoded_file(output_file, encoded_result)
    print(f"编码后的数据已保存到文件 {output_file}")

except Exception as e:
    print(f"发生错误: {e}")