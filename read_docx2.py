import docx2txt
import sys
import io

# 设置输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def extract_text_from_docx(file_path):
    try:
        print("正在使用docx2txt提取文本...")
        text = docx2txt.process(file_path)
        print(f"提取的文本长度: {len(text)}")
        print("提取的文本内容:")
        print("=" * 50)
        print(text)
        print("=" * 50)

        # 保存文本到文件
        with open('resume_text.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("文本已保存到resume_text.txt")

        return text
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    file_path = r"C:\Users\霍冠华\Desktop\简历\芦若禾简历.docx"
    extract_text_from_docx(file_path)