import pandas as pd
import numpy as np

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

print(f"原始数据行数: {len(df)}")
print(f"最高价列空值数量: {df['最高价'].isna().sum()}")
print(f"最低价列空值数量: {df['最低价'].isna().sum()}")

# 删除有空值的行（2行）
df = df.dropna(subset=['最高价', '最低价'])
print(f"删除空值后数据行数: {len(df)}")

# 按起点和目的地分组
grouped = df.groupby(['起点', '目的地'])

# 对每个分组：
# 1. 找出绝对最高价（所有订单中最高价的最大值）
# 2. 找出绝对最低价（所有订单中最低价的最小值）
# 3. 计算新平均值 = (绝对最高价 + 绝对最低价) / 2
# 4. 所有价格四舍五入到2位小数

result_data = []
for (起点, 目的地), group in grouped:
    绝对最高价 = group['最高价'].max()
    绝对最低价 = group['最低价'].min()
    新平均值 = (绝对最高价 + 绝对最低价) / 2

    # 四舍五入到2位小数
    绝对最高价 = round(绝对最高价, 2)
    绝对最低价 = round(绝对最低价, 2)
    新平均值 = round(新平均值, 2)

    result_data.append({
        '起点': 起点,
        '目的地': 目的地,
        '最高价': 绝对最高价,
        '最低价': 绝对最低价,
        '平均价': 新平均值
    })

# 创建新的DataFrame
result_df = pd.DataFrame(result_data)

print(f"\n处理后数据行数: {len(result_df)}")
print(f"合并后删除的重复订单: {len(df) - len(result_df)} 条")
print("\n前15行处理后的数据:")
print(result_df.head(15).to_string())

# 保存结果
output_file = r'C:\Users\霍冠华\Documents\trae_projects\claude code\运价处理后.xlsx'

# 创建Excel写入器
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 写入标题行
    title_df = pd.DataFrame([['2025年12月陕西运出煤炭、煤路线运价数据（处理后）']])
    title_df.to_excel(writer, index=False, header=False, startrow=0)

    # 写入列名
    columns_df = pd.DataFrame([['起点', '目的地', '最高价（元/吨）', '最低价（元/吨）', '平均价（元/吨）']])
    columns_df.to_excel(writer, index=False, header=False, startrow=1)

    # 写入数据
    result_df.to_excel(writer, index=False, header=False, startrow=2)

print(f"\n处理完成！结果已保存到: {output_file}")
print(f"共保留 {len(result_df)} 条不重复的路线数据")
print(f"所有价格已四舍五入到2位小数")