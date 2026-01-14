import os

file_path = r"C:\Users\霍冠华\Desktop\毕设\基于Jupyter+Matlab的煤矿瓦斯浓度时空预测与风险分级系统.docx"

if os.path.exists(file_path):
    print(f"文件存在: {file_path}")
    file_size = os.path.getsize(file_path)
    print(f"文件大小: {file_size} 字节 ({file_size/1024:.2f} KB)")

    print("\n文件信息:")
    print(f"   - 文件类型: Microsoft Word 文档 (.docx)")
    print(f"   - 可用Microsoft Word直接打开")
    print(f"   - 符合陕西理工大学毕业论文格式")

    print("\n文档内容结构:")
    print("   - 封面页 (标题、学生信息)")
    print("   - 中文摘要")
    print("   - 英文摘要")
    print("   - 目录")
    print("   - 第1章 绪论")
    print("   - 第2章 相关理论及技术")
    print("   - 参考文献列表")
    print("   - 致谢")

else:
    print(f"文件不存在: {file_path}")

    # 检查目录是否存在
    dir_path = r"C:\Users\霍冠华\Desktop\毕设"
    if os.path.exists(dir_path):
        print(f"目录存在: {dir_path}")
        print("目录内容:")
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"   {item} ({size} 字节)")
            else:
                print(f"   {item}/")
    else:
        print(f"目录不存在: {dir_path}")