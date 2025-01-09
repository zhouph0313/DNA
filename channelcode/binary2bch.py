import numpy as np
from math import log2, ceil


def generateBchMatrixAndSave(n, k, filename):
    """
    手动生成 BCH 生成矩阵并保存到 txt 文件。

    参数:
    n (int): 码字长度（通常是 2^m - 1）。
    k (int): 信息位长度（BCH 编码提供的有效信息位）。
    filename (str): 保存生成矩阵的文件名。
    """
    # BCH 编码生成矩阵的构造
    m = ceil(log2(n + 1))  # 计算 m 使得 n <= 2^m - 1
    if n != 2 ** m - 1:
        raise ValueError(f"n 必须是 2^m - 1，当前 n = {n}, m = {m} 不满足条件")

    # 生成循环码的生成多项式
    g = [1]  # 这里简单生成一个固定的生成多项式（如实际需求复杂，可以扩展）
    for _ in range(n - k):
        g = np.convolve(g, [1, 1]) % 2  # 用二元系数计算生成多项式

    # 生成矩阵 G 的构造
    G = np.zeros((k, n), dtype=int)
    for i in range(k):
        G[i, i:i + len(g)] = g  # 每行循环右移生成矩阵

    # 保存生成矩阵到文件
    np.savetxt(filename, G, fmt='%d')
    print(f"BCH 生成矩阵已保存到 {filename}")


def bchEncodeFromFile(inputFile, outputFile, matrixFile, k):
    """
    使用 BCH 矩阵对文件中的二进制数据进行编码，并保存到新文件。

    参数:
    inputFile (str): 包含原始二进制数据的文件。
    outputFile (str): 保存编码后数据的文件。
    matrixFile (str): 包含 BCH 生成矩阵的文件。
    k (int): 信息位长度。
    """
    # 从文件加载生成矩阵
    G = np.loadtxt(matrixFile, dtype=int)
    n = G.shape[1]  # 获取码字长度

    # 读取输入文件中的二进制数据
    with open(inputFile, "r") as file:
        content = file.read().strip()
    binary_data = np.array([int(bit) for bit in content], dtype=int)

    # 取长度为 k 的整数倍的部分
    original_length = len(binary_data)
    valid_length = (original_length // k) * k
    if valid_length < original_length:
        print(f"输入数据长度 {original_length} 不是 {k} 的倍数，"
              f"将截取前 {valid_length} 位用于编码。")
    binary_data = binary_data[:valid_length]

    # 对二进制数据分块编码
    encoded_data = []
    for i in range(0, len(binary_data), k):
        message_block = binary_data[i:i + k]
        encoded_block = np.mod(np.dot(message_block, G), 2)
        encoded_data.extend(encoded_block)

    # 保存编码后的数据到文件
    with open(outputFile, "w") as file:
        file.write("".join(map(str, encoded_data)))
    print(f"编码后的数据已保存到 {outputFile}")


# 示例运行
try:
    # 生成生成矩阵
    matrixFile = "C:/Users/zhoupenghua/Desktop/bch_matrix3.txt"
    generateBchMatrixAndSave(n=63, k=16, filename=matrixFile)  # BCH(15, 11)

    # 使用生成的矩阵进行编码
    inputFile = "C:/Users/zhoupenghua/Desktop/binarymessage.txt"
    outputFile = "C:/Users/zhoupenghua/Desktop/bch_binary3.txt"
    bchEncodeFromFile(inputFile=inputFile, outputFile=outputFile, matrixFile=matrixFile, k=16)
except Exception as e:
    print(f"程序运行时发生错误: {e}")
