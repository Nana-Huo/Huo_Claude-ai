"""
根据论文需求选择合适的变量
功能：手动选择符合论文要求的变量
"""

import pandas as pd
import numpy as np

print("="*60)
print("根据论文需求选择合适的变量")
print("="*60)

# 1. 读取原始数据
print("\n1. 读取原始数据...")
data = pd.read_excel(r'C:\Users\霍冠华\AppData\Local\Programs\Python\Python313\python.exe' -c "import pandas as pd; df = pd.read_excel(r'C:\Users\霍冠华\Documents\trae_projects\claude code\原始数据.xlsx'); print(df.columns.tolist())")
target_column = data.columns[-1]  # 使用最后一列作为目标变量

print(f"原始数据形状: {data.shape}")
print(f"目标变量: {target_column}")

# 2. 根据论文需求选择变量
print("\n2. 根据论文需求选择变量...")

# 论文要求：
# 1. 区域特征（饮食文化、生活习惯、地理环境）
# 2. 风险因素分析（卡方检验、Logistic回归、随机森林）
# 3. 空间分布图（需要有区域信息）

# 手动选择符合论文需求的变量
selected_features = {
    '人口学特征': [
        '4.您的性别',
        '5.您的身高是多少cm？',
        '6.您的体重是多少Kg',
        '7.年龄',
    ],
    '饮食文化特征': [
        '49.您对辣的食物耐受程度？',
        '10.您有兄弟姐妹吗？',
        '16.您一般一顿饭食量多少？（5分制）',
        '48.辛辣调味品使用频率-单位（次/周、顿）',
        '48.酒',
        '48.茶',
    ],
    '生活习惯特征': [
        '11.您一般是否经常运动？',
        '12.您是否每天固定时间睡觉？',
        '13.您平时饮食喝水吗？还是其他？',
        '14.您平时饮食喝水是否使用凉开水？',
        '15.您的家庭在几个月内有一顿大餐？',
        '17.您一般是否经常熬夜？',
        '18.您是否经常熬夜后吃夜宵？',
        '19.您是否经常吃油炸食品？',
        '20.您是否经常吃外卖？',
        '30.您觉得需要经常上厕所的。',
        '30.您觉得生活压力大。',
        '30.您最近一段时间睡眠质量好。',
        '30.您有胃不舒服。',
        '30.您觉得身体变差了。',
        '30.您觉得身体变胖了。',
    ],
    '健康状况特征': [
        '41.您的父母或祖父母是否患有以下疾病或慢性高血压？',
        '41.糖尿病',
        '41.高血压',
        '41.胆结石',
        '41.高血脂',
        '41.脂肪肝',
        '44.您的母亲总胆固醇值是多少？（如不知道为0）',
        '45.您的总胆固醇值是多少？（如不知道为0）',
        '46.您的甘油三酯值是多少？（如不知道为0）',
        '47.您的低密度脂蛋白值是多少？（如不知道为0）',
        '51.使用抗生素药物',
    ],
    '心理特征': [
        '30.您情绪低落，抑郁。',
        '30.您容易焦虑。',
        '30.您感觉身体疲劳。',
        '30.您睡眠质量差。',
        '30.您感到胃痛。',
        '30.您感到腹胀。',
    ],
    '用药史特征': [
        '24.您有没有正在服用或在最近服用过的某种药物？',
        '25.您目前在服用什么药物？',
    ],
    '临床症状特征': [
        '64.右上腹痛是否常在进食油腻食物之后',
        '65.您是否有过右上腹疼痛？',
    ]
}

# 3. 验证选择的特征是否存在
print("\n3. 验证选择的特征...")
all_selected_features = []
valid_features = {}

for category, features in selected_features.items():
    valid_features[category] = []
    for feature in features:
        if feature in data.columns:
            valid_features[category].append(feature)
            all_selected_features.append(feature)
            print(f"[OK] {category}: {feature}")
        else:
            print(f"[X] {category}: {feature} (不存在)")

print(f"\n总共选择 {len(all_selected_features)} 个特征")

# 4. 创建筛选后的数据集
print("\n4. 创建筛选后的数据集...")
selected_data = data[all_selected_features + [target_column]].copy()

print(f"筛选后数据形状: {selected_data.shape}")

# 5. 数据预处理
print("\n5. 数据预处理...")

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
from sklearn.preprocessing import LabelEncoder

for column in selected_data.columns:
    if column != target_column and selected_data[column].dtype in ['object', 'category']:
        le = LabelEncoder()
        selected_data[column] = le.fit_transform(selected_data[column].astype(str))

print(f"预处理后数据形状: {selected_data.shape}")

# 6. 保存筛选后的数据
output_dir = r"C:\Users\霍冠华\Documents\trae_projects\claude code"
论文筛选数据 = f"{output_dir}/论文筛选数据.csv"
selected_data.to_csv(论文筛选数据, index=False, encoding='utf-8-sig')
print(f"\n已保存: {论文筛选数据.csv}")

# 7. 保存特征列表
feature_list_file = f"{output_dir}/论文筛选特征列表.txt"
with open(feature_list_file, 'w', encoding='utf-8') as f:
    f.write("根据论文需求选择的变量\n")
    f.write("="*60 + "\n\n")
    
    for category, features in valid_features.items():
        f.write(f"{category}:\n")
        for i, feature in enumerate(features, 1):
            f.write(f"  {i}. {feature}\n")
        f.write("\n")

print(f"已保存: {feature_list_file}")

# 8. 生成报告
report = f"""
根据论文需求选择变量报告
{'='*60}

一、论文需求分析
1. 区域性疾病风险分析综述
   - 需要区域特征（饮食文化、生活习惯、地理环境）
   - 需要能够绘制空间分布图的变量

2. 疾病风险分析方法综述
   - 传统统计学方法：卡方检验、单因素分析、多因素Logistic回归
   - 机器学习方法：决策树、随机森林、支持向量机SVM

3. 问卷数据预处理与实证分析
   - 数据预处理：变量分类、数据清洗、编码转换
   - 风险因素分析：卡方检验、Logistic回归、随机森林
   - 结果可视化与建议：空间分布图、区域特异性分析

二、变量选择原则
1. 符合论文研究主题（区域性疾病风险调研）
2. 能够反映区域特征（饮食文化、生活习惯）
3. 适合统计分析和机器学习
4. 能够绘制空间分布图（如果有区域信息）
5. 样本量与特征数比例合理（约10:1）

三、选择的变量
"""

for category, features in valid_features.items():
    report += f"\n{category} ({len(features)}个):\n"
    for i, feature in enumerate(features, 1):
        report += f"  {i}. {feature}\n"

report += f"""
四、数据信息
- 原始数据形状: {data.shape}
- 原始特征数量: {len(data.columns) - 1}
- 筛选后数据形状: {selected_data.shape}
- 筛选后特征数量: {len(all_selected_features)}
- 样本量: {len(selected_data)}
- 样本量/特征数: {len(selected_data) / len(all_selected_features):.1f}:1

五、下一步分析
1. 卡方检验：筛选与胆结石发病相关的潜在风险因素
2. Logistic回归：识别独立风险因素
3. 随机森林：风险因素重要性排序
4. 方法比较：对比三种方法的分析结果
5. 概率预测：进行风险分层
6. 结果可视化：绘制空间分布图

六、输出文件
1. 论文筛选数据.csv - 筛选后的数据
2. 论文筛选特征列表.txt - 特征列表
"""

report_file = f"{output_dir}/论文变量选择报告.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"已保存: {report_file}")

print("\n" + "="*60)
print("变量选择完成！")
print("="*60)
print(f"选择了 {len(all_selected_features)} 个符合论文需求的变量")
print("样本量/特征数比例: {:.1f}:1".format(len(selected_data) / len(all_selected_features)))
