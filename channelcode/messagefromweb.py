'''with open('C:/Users/zhoupenghua/Desktop/output.txt', 'r', encoding='utf-8') as infile:
    content = infile.read()
print(len(content)'''
import requests
from bs4 import BeautifulSoup
def text_to_binary(text):
    """
    将字符串转换为二进制表示
    :param text: 输入字符串
    :return: 字符串对应的二进制表示
    """
    return ''.join(format(ord(char), '08b') for char in text)

def fetch_web_content(url):
    """
    爬取指定网页的文本内容并转换为二进制。
    :param url: 网页 URL
    :return: 网页内容的二进制表示
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # 提取网页主要内容
        text = soup.get_text()
        return text
    except Exception as e:
        print(f"爬取失败: {e}")
        return None

# 示例使用
url = "https://en.wikipedia.org/wiki/Tree"
txt = fetch_web_content(url)
print(len(txt))
if txt:
    with open("C:/Users/zhoupenghua/Desktop/web_output_test.txt", 'a', encoding='utf-8') as file:
        file.write(txt)
