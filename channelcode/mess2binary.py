def text_to_binary(text):
    """
    将字符串转换为二进制表示
    :param text: 输入字符串
    :return: 字符串对应的二进制表示
    """
    return ''.join(format(ord(char), '08b') for char in text)


def convert_file_to_binary(input_file, output_file):
    """
    从输入文件读取内容，转换为二进制，并保存到输出文件
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    """
    try:
        # 读取输入文件内容
        with open(input_file, 'r', encoding='utf-8') as infile:
            content = infile.read()
        print(len(content))
        # 转换为二进制
        binary_content = text_to_binary(content)
        print(len(binary_content))
        # 保存到输出文件
        with open(output_file, 'a', encoding='utf-8') as outfile:
            outfile.write(binary_content)

        print(f"转换完成！二进制内容已保存到 {output_file}")
    except Exception as e:
        print(f"发生错误: {e}")


# 示例使用
input_path = "C:/Users/zhoupenghua/Desktop/web_output.txt"  # 输入文件路径
output_path = "C:/Users/zhoupenghua/Desktop/output.txt"  # 输出文件路径
convert_file_to_binary(input_path, output_path)
