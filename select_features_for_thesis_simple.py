"""
根据论文需求选择变量（简化版）
功能：使用前50个变量，选择符合论文需求的变量
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("="*60)
print("根据论文需求选择变量")
print("="*60)

# 1. 读取原始数据
print("\n1. 读取原始数据...")
data = pd.read_excel(r'C:\Users\霍冠华\Documents\trae_projects\claude code\原始数据.xlsx')
target_column = data.columns[1]  # 使用第2列作为目标变量

print(f"原始数据形状: {data.shape}")
print(f"目标变量: {target_column}")

# 2. 选择前50个变量（排除序号和泄露特征）
print("\n2. 选择符合论文需求的变量...")

# 手动选择符合论文需求的变量（基于前50个变量）
selected_feature_indices = [
    5,   # 4.您的性别
    6,   # 5.您的身高是多少cm？
    7,   # 6.您的体重是多少Kg
    8,   # 7.年龄
    11,  # 10.您有兄弟姐妹吗？
    12,  # 11.您一般是否经常运动？
    13,  # 12.您是否每天固定时间睡觉？
    14,  # 13.您平时饮食喝水吗？还是其他？
    16,  # 15.您的家庭在几个月内有一顿大餐？
    17,  # 16.您一般一顿饭食量多少？（5分制）
    18,  # 17.您一般是否经常熬夜？
    19,  # 18.您是否经常熬夜后吃夜宵？
    20,  # 19.您是否经常吃油炸食品？
    21,  # 20.您是否经常吃外卖？
    22,  # 21.您是否经常吃辛辣？
    23,  # 22.您是否经常吃过烫？
    24,  # 23.您是否经常吃冷饮？
    25,  # 24.您有没有正在服用或在最近服用过的某种药物？
    26,  # 25.您目前在服用什么药物？
    27,  # 26.您最近一个月睡眠质量如何？
    28,  # 27.您一般一次睡眠时间是多少？
    29,  # 28.您最近睡眠是否充足？
    30,  # 29.您的木工活是多少岁？
    31,  # 30.您平时是否经常感到胃不舒服？
    32,  # 30.您是否觉得生活压力大？
    33,  # 30.您最近一段时间睡眠质量好。
    34,  # 30.您有胃不舒服。
    35,  # 30.您觉得身体变差了。
    36,  # 30.您觉得身体变胖了。
    37,  # 30.您情绪低落，抑郁。
    38,  # 30.您容易焦虑。
    39,  # 30.您感觉身体疲劳。
    40,  # 30.您睡眠质量差。
    41,  # 30.您感到胃痛。
    42,  # 30.您感到腹胀。
    43,  # 30.您觉得身体变冷了。
    44,  # 30.您总是感到饿。
    45,  # 30.您觉得身体变暖了。
    46,  # 30.您最近一段时间睡眠质量好。
    47,  # 30.您觉得需要经常上厕所的。
    48,  # 30.您最近一段时间睡眠质量好。
    49,  # 30.您觉得经常需要上厕所的。
    50,  # 31.您最近一个月是否影响睡眠质量？睡眠障碍、失眠、30天内经常失眠
]

selected_features = [data.columns[i] for i in selected_feature_indices if i < len(data.columns)]

print(f"选择了 {len(selected_features)} 个特征:")
for i, feature in enumerate(selected_features, 1):
    print(f"  {i}. {feature}")

# 3. 创建筛选后的数据集
print("\n3. 创建筛选后的数据集...")
selected_data = data[selected_features + [target_column]].copy()

print(f"筛选后数据形状: {selected_data.shape}")

# 4. 数据预处理
print("\n4. 数据预处理...")

# 处理缺失值
threshold = len(selected_data.columns) * 0.5
selected_data = selected_data.dropna(thresh=threshold)

for column in selected_data.columns:
    if column != target_column:
        if selected_data[column].dtype in ['object', 'category']:
            mode_value = selected_data[column].mode()
            if len(mode_value) > 0:
                selected_data[column].fillna(mode_value[0], inplace=True)
        else:
            selected_data[column].fillna(selected_data[column].mean(), inplace=True)

# 编码分类变量
for column in selected_data.columns:
    if column != target_column and selected_data[column].dtype in ['object', 'category']:
        le = LabelEncoder()
        selected_data[column] = le.fit_transform(selected_data[column].astype(str))

print(f"预处理后数据形状: {selected_data.shape}")

# 5. 保存筛选后的数据
output_dir = r"C:\Users\霍冠华\Documents\trae_projects\claude code"
论文筛选数据 = f"{output_dir}/论文筛选数据.csv"
selected_data.to_csv(论文筛选数据, index=False, encoding='utf-8-sig')
print(f"\n已保存: {论文筛选数据.csv}")

# 6. 保存特征列表
feature_list_file = f"{output_dir}/论文筛选特征列表.txt"
with open(feature_list_file, 'w', encoding='utf-8') as f:
    f.write("根据论文需求选择的变量\n")
    f.write("="*60 + "\n\n")
    f.write("选择的变量:\n")
    for i, feature in enumerate(selected_features, 1):
        f.write(f"{i}. {feature}\n")

print(f"已保存: {feature_list_file}")

print("\n" + "="*60)
print("变量选择完成！")
print("="*60)
print(f"选择了 {len(selected_features)} 个符合论文需求的变量")
print(f"样本量: {len(selected_data)}")
print(f"样本量/特征数: {len(selected_data) / len(selected_features):.1f}:1")