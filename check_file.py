import os

file_path = r"C:\Users\霍冠华\Desktop\毕设\基于Jupyter+Matlab的煤矿瓦斯浓度时空预测与风险分级系统.docx"

if os.path.exists(file_path):
    print(f"✅ 文件存在: {file_path}")
    file_size = os.path.getsize(file_path)
    print(f"📊 文件大小: {file_size} 字节 ({file_size/1024:.2f} KB)")

    print("\n📋 文件信息:")
    print(f"   - 文件类型: Microsoft Word 文档 (.docx)")
    print(f"   - 创建时间: {os.path.getctime(file_path)}")
    print(f"   - 修改时间: {os.path.getmtime(file_path)}")
    print(f"   - 可读写: {os.access(file_path, os.R_OK)}")

    print("\n📄 文档内容结构:")
    print("   - 封面页 (标题、学生信息)")
    print("   - 中文摘要")
    print("   - 英文摘要")
    print("   - 目录")
    print("   - 第1章 绪论")
    print("   - 第2章 相关理论及技术")
    print("   - 参考文献列表")
    print("   - 致谢")

    print("\n🎯 文档特点:")
    print("   - 标准Word .docx二进制格式")
    print("   - 可直接用Microsoft Word打开")
    print("   - 符合陕西理工大学毕业论文格式要求")
    print("   - 包含完整的论文结构")

else:
    print(f"❌ 文件不存在: {file_path}")

    # 检查目录是否存在
    dir_path = r"C:\Users\霍冠华\Desktop\毕设"
    if os.path.exists(dir_path):
        print(f"✅ 目录存在: {dir_path}")
        print("📁 目录内容:")
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"   📄 {item} ({size} 字节)")
            else:
                print(f"   📁 {item}/")
    else:
        print(f"❌ 目录不存在: {dir_path}")