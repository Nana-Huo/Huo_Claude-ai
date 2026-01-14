import os
import pandas as pd

# 搜索路径
search_path = r"C:\Users\霍冠华\Desktop\zczy工作留痕"
target_plate = "宁B76107"

found_files = []

# 遍历所有文件
for root, dirs, files in os.walk(search_path):
    for file in files:
        if file.endswith(('.xls', '.xlsx')):
            file_path = os.path.join(root, file)
            try:
                # 读取Excel文件
                df = pd.read_excel(file_path)
                # 搜索目标车牌
                if target_plate in df.values.astype(str):
                    found_files.append(file_path)
                    print(f"找到: {file_path}")
            except Exception as e:
                pass

if not found_files:
    print(f"未找到包含 {target_plate} 的文件")
else:
    print(f"\n共找到 {len(found_files)} 个文件")