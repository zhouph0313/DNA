import numpy as np

def generateLdpcMatrixAndSave(n, k, filename):
    """
    手动生成 LDPC 生成矩阵并保存到 txt 文件。
    参数:
    n (int): 码字长度。
    k (int): 信息位长度。
    filename (str): 保存生成矩阵的文件名。
    """
    # 简单实现：使用随机矩阵 + 单位矩阵构造生成矩阵
    P = np.random.randint(0, 2, size=(k, n-k))  # 随机矩阵
    I = np.eye(k, dtype=int)  # 单位矩阵
    G = np.hstack((I, P))  # 合成生成矩阵

    # 保存生成矩阵到文件
    np.savetxt(filename, G, fmt='%d')
    print(f"LDPC 生成矩阵已保存到 {filename}")


def ldpcEncodeFromFile(inputFile, outputFile, matrixFile, k):
    """
    使用 LDPC 矩阵对文件中的二进制数据进行编码，并保存到新文件。
    参数:
    inputFile (str): 包含原始二进制数据的文件。
    outputFile (str): 保存编码后数据的文件。
    matrixFile (str): 包含 LDPC 生成矩阵的文件。
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
        message_block = binary_data[i:i+k]
        encoded_block = np.mod(np.dot(message_block, G), 2)
        encoded_data.extend(encoded_block)

    # 保存编码后的数据到文件
    with open(outputFile, "w") as file:
        file.write("".join(map(str, encoded_data)))
    print(f"编码后的数据已保存到 {outputFile}")


# 示例运行
try:
    # 生成生成矩阵
    matrixFile = "C:/Users/zhoupenghua/Desktop/ldpc_matrix_5.txt"
    generateLdpcMatrixAndSave(n=121, k=50, filename=matrixFile)

    # 使用生成的矩阵进行编码
    inputFile = "C:/Users/zhoupenghua/Desktop/binarymessage.txt"
    outputFile = "C:/Users/zhoupenghua/Desktop/ldpc_binary_5.txt"
    ldpcEncodeFromFile(inputFile=inputFile, outputFile=outputFile, matrixFile=matrixFile, k=50)
except Exception as e:
    print(f"程序运行时发生错误: {e}")


