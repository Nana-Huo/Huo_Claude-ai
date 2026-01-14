import pandas as pd

# 读取原始数据
file_path = r'C:\Users\霍冠华\Documents\trae_projects\claude code\运价.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# 跳过第一行（标题行），从第二行开始读取
df.columns = df.iloc[0]  # 使用第二行作为列名
df = df.drop(df.columns[0], axis=1)  # 删除序号列
df.columns = ['起点', '目的地', '原平均价', '最高价', '最低价']
df = df.iloc[1:].reset_index(drop=True)

# 转换数据类型
df['最高价'] = pd.to_numeric(df['最高价'], errors='coerce')
df['最低价'] = pd.to_numeric(df['最低价'], errors='coerce')

print("原始数据检查:")
print(f"总行数: {len(df)}")
print(f"\n最高价列空值数量: {df['最高价'].isna().sum()}")
print(f"最低价列空值数量: {df['最低价'].isna().sum()}")

print("\n最高价列数据类型和样本:")
print(df['最高价'].dtype)
print(df['最高价'].dropna().head(20))

print("\n最低价列数据类型和样本:")
print(df['最低价'].dtype)
print(df['最低价'].dropna().head(20))

print("\n有小数的最高价样本:")
print(df[df['最高价'] != df['最高价'].round(0)].head(10))

print("\n有小数的最低价样本:")
print(df[df['最低价'] != df['最低价'].round(0)].head(10))

# 检查是否有相同的起点和终点
print("\n\n检查重复的起点-终点组合:")
duplicates = df[df.duplicated(['起点', '目的地'], keep=False)]
print(f"有重复的路线数量: {len(duplicates)}")
print("\n重复路线示例:")
print(duplicates.head(20))