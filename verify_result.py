import pandas as pd

# 读取处理后的文件
output_file = r'C:\Users\霍冠华\Documents\trae_projects\claude code\运价处理后.xlsx'

# 读取Excel文件
df = pd.read_excel(output_file, engine='openpyxl', header=1)  # 从第二行开始读取列名

print("处理后的文件数据:")
print(f"数据行数: {len(df)}")
print(f"列名: {df.columns.tolist()}")
print("\n前15行数据:")
print(df.head(15).to_string())
print("\n数据统计:")
print(df.describe().to_string())