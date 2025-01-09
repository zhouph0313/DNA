def transformBinaryData(inputFile, outputFile):
    """
    将二进制数据进行转换，00->0, 01->1, 10->2, 11->3，并保存到新的文件中。
    如果输入数据长度是奇数，则截取最大偶数长度进行转换。

    参数:
    inputFile (str): 输入包含原始二进制数据的文件。
    outputFile (str): 输出转换后数据的文件。
    """
    # 读取二进制数据
    with open(inputFile, 'r') as file:
        content = file.read().strip()  # 去除文件开头和结尾的空白字符

    # 检查输入数据的长度是否是偶数
    if len(content) % 2 != 0:
        print(f"输入数据长度 {len(content)} 是奇数，将截取前 {len(content) - 1} 位进行转换。")
        content = content[:-1]  # 截取掉最后一位，确保长度为偶数

    # 定义二进制对到数字的映射
    binary_to_int = {'00': '0', '01': '1', '10': '2', '11': '3'}

    # 处理每两个二进制位
    transformed_data = []
    for i in range(0, len(content), 2):
        binary_pair = content[i:i + 2]
        if binary_pair in binary_to_int:
            transformed_data.append(binary_to_int[binary_pair])
        else:
            print(f"遇到不合法的二进制对: {binary_pair}")
            return

    # 将转换后的数据写入输出文件
    with open(outputFile, 'w') as file:
        file.write("".join(transformed_data))  # 写入字符串形式的数字

    print(f"转换后的数据已保存到 {outputFile}")


input = "C:/Users/zhoupenghua/Desktop/conveencode3.txt"
output = "C:/Users/zhoupenghua/Desktop/convedna3.txt"
# 示例运行


transformBinaryData(input, output)


