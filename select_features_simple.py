import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("根据论文需求选择变量")

# 读取原始数据
data = pd.read_excel(r'C:\Users\霍冠华\Documents\trae_projects\claude code\原始数据.xlsx')
target_column = data.columns[1]

print(f"原始数据形状: {data.shape}")
print(f"目标变量: {target_column}")

# 选择前50个变量中的关键变量
selected_indices = [5,6,7,8,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
selected_features = [data.columns[i] for i in selected_indices if i < len(data.columns)]

print(f"选择了 {len(selected_features)} 个特征")

# 创建筛选后的数据
selected_data = data[selected_features + [target_column]].copy()

# 数据预处理
threshold = len(selected_data.columns) * 0.5
selected_data = selected_data.dropna(thresh=threshold)

for column in selected_data.columns:
    if column != target_column:
        if selected_data[column].dtype in ['object', 'category']:
            mode_value = selected_data[column].mode()
            if len(mode_value) > 0:
                selected_data[column] = selected_data[column].fillna(mode_value[0])
        else:
            selected_data[column] = selected_data[column].fillna(selected_data[column].mean())

# 编码分类变量
for column in selected_data.columns:
    if column != target_column and selected_data[column].dtype in ['object', 'category']:
        le = LabelEncoder()
        selected_data[column] = le.fit_transform(selected_data[column].astype(str))

print(f"预处理后数据形状: {selected_data.shape}")

# 保存数据
output_dir = r"C:\Users\霍冠华\Documents\trae_projects\claude code"
selected_data.to_csv(f"{output_dir}/论文筛选数据.csv", index=False, encoding='utf-8-sig')

# 保存特征列表
with open(f"{output_dir}/论文筛选特征列表.txt", 'w', encoding='utf-8') as f:
    f.write("选择的变量:\n")
    for i, feature in enumerate(selected_features, 1):
        f.write(f"{i}. {feature}\n")

print(f"样本量/特征数: {len(selected_data) / len(selected_features):.1f}:1")
print("变量选择完成!")