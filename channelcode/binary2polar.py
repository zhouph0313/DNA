import numpy as np

def generatePolarMatrixAndSave(n, filename):
    """
    生成 Polar 码的生成矩阵并保存到 txt 文件。

    参数:
    n (int): Polar 码长度（必须为 2 的幂）。
    filename (str): 保存生成矩阵的文件名。
    """
    # 验证 n 是否为 2 的幂
    if not (n > 0 and (n & (n - 1)) == 0):
        raise ValueError(f"n 必须为 2 的幂，当前 n = {n} 不符合条件。")

    # 初始化生成矩阵 F
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F

    # 构造生成矩阵 G = F ⊗ F ⊗ ... ⊗ F（Kronecker 积）
    m = int(np.log2(n))
    for _ in range(m - 1):
        G = np.kron(G, F)

    # 保存生成矩阵到文件
    np.savetxt(filename, G, fmt='%d')
    print(f"Polar 生成矩阵已保存到 {filename}")


def polarEncodeFromFile(inputFile, outputFile, matrixFile, k):
    """
    使用 Polar 生成矩阵对文件中的二进制数据进行编码，并保存到新文件。

    参数:
    inputFile (str): 包含原始二进制数据的文件。
    outputFile (str): 保存编码后数据的文件。
    matrixFile (str): 包含 Polar 生成矩阵的文件。
    k (int): 信息位长度（高可靠信道的个数）。
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

    # 确定可靠信道的索引（简单假设选择前 k 个信道为可靠信道）
    frozen_indices = np.zeros(n, dtype=int)
    frozen_indices[:k] = 1  # 高可靠信道
    frozen_indices = np.argsort(frozen_indices)[::-1]  # 高可靠信道排在前面

    # 编码过程
    encoded_data = []
    for i in range(0, len(binary_data), k):
        message_block = binary_data[i:i+k]
        u = np.zeros(n, dtype=int)
        u[frozen_indices[:k]] = message_block  # 信息位放入高可靠信道
        encoded_block = np.mod(np.dot(u, G), 2)
        encoded_data.extend(encoded_block)

    # 保存编码后的数据到文件
    with open(outputFile, "w") as file:
        file.write("".join(map(str, encoded_data)))
    print(f"编码后的数据已保存到 {outputFile}")


# 示例运行
try:
    # 生成生成矩阵
    matrixFile = "C:/Users/zhoupenghua/Desktop/polar_matrix3.txt"
    generatePolarMatrixAndSave(n=64, filename=matrixFile)  # Polar 码长度 n =

    # 使用生成的矩阵进行编码
    inputFile = "C:/Users/zhoupenghua/Desktop/binarymessage.txt"
    outputFile = "C:/Users/zhoupenghua/Desktop/polar_binary3.txt"
    polarEncodeFromFile(inputFile=inputFile, outputFile=outputFile, matrixFile=matrixFile, k=36)
except Exception as e:
    print(f"程序运行时发生错误: {e}")
