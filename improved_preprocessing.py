"""
改进数据预处理方法的分析
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings

warnings.filterwarnings('ignore')

print("="*60)
print("改进数据预处理方法的分析")
print("="*60)

output_dir = r"C:\Users\霍冠华\Documents\trae_projects\claude code"

# 1. 加载原始数据
print("\n1. 加载原始数据...")
data = pd.read_excel(f"{output_dir}/原始数据.xlsx")
print(f"原始数据形状: {data.shape}")

# 2. 获取列名
columns = data.columns.tolist()

# 3. 选择目标变量和特征
print("\n2. 选择特征...")
gallstone_col = columns[1]  # 1、您是否患有胆囊结石？

# 基础特征
age_col = columns[7]  # 年龄
sex_col = columns[4]  # 性别
height_col = columns[5]  # 身高
weight_col = columns[6]  # 体重
waist_col = columns[8]  # 腰围

# 饮食相关
spicy_freq_col = columns[20]  # 是否经常吃火锅？
pickle_times_col = columns[129]  # 腌制肉
chili_oil_meals_col = columns[21]  # 火锅锅底

# 生活习惯
smoke_col = columns[57]  # 吸烟
drink_col = None  # 饮酒（未找到）

# 地理环境
water_src_col = columns[10]  # 县区
altitude_col = None  # 海拔（未找到）
town_id_col = None  # 乡镇（未找到）

# 血脂
TC_col = None  # 总胆固醇（未找到）
TG_col = columns[45]  # 低密度脂蛋白

# 其他
abdominal_pain_col = columns[64]  # 腹痛是否在进食油腻或饮酒之后

# 构建特征列表
selected_features = [
    gallstone_col, age_col, sex_col, height_col, weight_col,
    waist_col, spicy_freq_col, pickle_times_col, chili_oil_meals_col,
    smoke_col, water_src_col, TG_col, abdominal_pain_col
]

# 过滤掉None值
selected_features = [f for f in selected_features if f is not None]

print(f"选择的特征数量: {len(selected_features)}")
print(f"特征列表:")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i}. {feat}")

# 4. 准备数据
print("\n3. 准备数据...")
data_selected = data[selected_features].copy()

# 转换目标变量
y = data_selected[gallstone_col].map({'没有': 0, '有': 1})

# 删除y中的NaN值
valid_indices = ~y.isna()
y = y[valid_indices]
data_selected = data_selected[valid_indices]

print(f"有效样本数: {len(y)}")
print(f"患病率: {y.mean():.2%}")

# 5. 计算BMI
if height_col and weight_col:
    try:
        data_selected[height_col] = pd.to_numeric(data_selected[height_col], errors='coerce')
        data_selected[weight_col] = pd.to_numeric(data_selected[weight_col], errors='coerce')
        data_selected['BMI'] = data_selected[weight_col] / ((data_selected[height_col] / 100) ** 2)
        print(f"  已计算BMI")
    except Exception as e:
        print(f"  计算BMI失败: {e}")

# 6. 分析每个特征的数据类型和分布
print("\n4. 分析特征数据类型和分布...")
for col in data_selected.columns:
    if col == gallstone_col:
        continue
    print(f"\n{col}:")
    print(f"  数据类型: {data_selected[col].dtype}")
    print(f"  唯一值数量: {data_selected[col].nunique()}")
    print(f"  缺失值数量: {data_selected[col].isna().sum()}")
    print(f"  缺失值比例: {data_selected[col].isna().sum() / len(data_selected) * 100:.2f}%")

    # 显示前10个值
    print(f"  前10个值: {data_selected[col].head(10).tolist()}")

    # 如果是数值型，显示统计信息
    if data_selected[col].dtype in ['int64', 'float64']:
        print(f"  最小值: {data_selected[col].min()}")
        print(f"  最大值: {data_selected[col].max()}")
        print(f"  均值: {data_selected[col].mean():.2f}")
        print(f"  中位数: {data_selected[col].median():.2f}")
    else:
        # 显示值计数
        value_counts = data_selected[col].value_counts()
        print(f"  值分布:")
        for val, count in value_counts.items():
            print(f"    {val}: {count} ({count/len(data_selected)*100:.1f}%)")

# 7. 区分数值型和分类型特征
print("\n5. 区分数值型和分类型特征...")

numeric_features = []
categorical_features = []

for col in data_selected.columns:
    if col == gallstone_col or col == height_col or col == weight_col:
        continue

    # 尝试转换为数值
    try:
        data_selected[col] = pd.to_numeric(data_selected[col], errors='coerce')
        # 如果转换后缺失值比例超过50%，认为是分类变量
        if data_selected[col].isna().sum() / len(data_selected) > 0.5:
            categorical_features.append(col)
        elif data_selected[col].nunique() < 10:  # 唯一值少于10个，可能是分类
            categorical_features.append(col)
        else:
            numeric_features.append(col)
    except:
        categorical_features.append(col)

print(f"\n数值型特征 ({len(numeric_features)}):")
for feat in numeric_features:
    print(f"  - {feat}")

print(f"\n分类型特征 ({len(categorical_features)}):")
for feat in categorical_features:
    print(f"  - {feat}")

# 8. 构建预处理管道
print("\n6. 构建预处理管道...")

# 数值型特征的处理：中位数填充缺失值 + 标准化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 分类型特征的处理：众数填充缺失值 + 独热编码
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 组合预处理步骤
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # 保留未指定的列
)

# 9. 准备特征矩阵
print("\n7. 准备特征矩阵...")
X = data_selected.drop(columns=[gallstone_col, height_col, weight_col])

print(f"特征矩阵形状: {X.shape}")

# 10. 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")

# 11. 构建完整管道（预处理 + 模型）
print("\n8. 构建预测模型...")
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        oob_score=True
    ))
])

# 训练模型
model.fit(X_train, y_train)
print("模型训练完成")

# 12. 评估模型性能
print("\n9. 评估模型性能...")
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_prob)

print(f"\n测试集性能:")
print(f"  准确率: {test_accuracy:.4f}")
print(f"  AUC: {test_auc:.4f}")
print(f"  OOB得分: {model.named_steps['classifier'].oob_score_:.4f}")

# 交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
cv_auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

print(f"\n5折交叉验证结果:")
print(f"  准确率: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"  AUC: {cv_auc_scores.mean():.4f} +/- {cv_auc_scores.std():.4f}")

# 13. 分类报告
print("\n10. 分类报告:")
print(classification_report(y_test, y_test_pred, target_names=['未患病', '患病']))

# 14. 获取特征重要性
print("\n11. 特征重要性分析...")

# 获取预处理后的特征名
preprocessor = model.named_steps['preprocessor']
feature_names = []

# 数值型特征名
feature_names.extend(numeric_features)

# 分类型特征名（独热编码后）
cat_transformer = preprocessor.named_transformers_['cat']
onehot = cat_transformer.named_steps['onehot']
for cat_feature in categorical_features:
    # 获取该分类特征的所有可能值
    unique_values = data_selected[cat_feature].dropna().unique()
    for val in unique_values:
        feature_names.append(f"{cat_feature}_{val}")

# 获取特征重要性
importances = model.named_steps['classifier'].feature_importances_

# 创建DataFrame
feature_importance = pd.DataFrame({
    '特征': feature_names,
    '重要性': importances
}).sort_values('重要性', ascending=False)

print(f"\n所有特征的重要性排序:")
for i, row in feature_importance.head(20).iterrows():
    print(f"  {i+1}. {row['特征']}: {row['重要性']:.4f}")

# 15. 概率预测和风险分层
print("\n12. 概率预测和风险分层...")
all_prob = model.predict_proba(X)[:, 1]

risk_levels = []
for prob in all_prob:
    if prob < 0.33:
        risk_levels.append('低风险')
    elif prob < 0.67:
        risk_levels.append('中风险')
    else:
        risk_levels.append('高风险')

results_df = pd.DataFrame({
    '实际值': y.values,
    '患病概率': all_prob,
    '风险等级': risk_levels
})

print(f"\n风险等级分布:")
risk_distribution = results_df['风险等级'].value_counts().sort_index()
for level, count in risk_distribution.items():
    percentage = count / len(results_df) * 100
    print(f"  {level}: {count}人 ({percentage:.1f}%)")

print(f"\n各风险等级的实际患病率:")
for level in ['低风险', '中风险', '高风险']:
    level_data = results_df[results_df['风险等级'] == level]
    if len(level_data) > 0:
        actual_rate = level_data['实际值'].mean() * 100
        print(f"  {level}: {actual_rate:.1f}% (样本数: {len(level_data)})")

# 16. 保存结果
print("\n13. 保存结果...")

# 保存特征重要性
importance_file = f"{output_dir}/改进预处理_特征重要性.csv"
feature_importance.to_csv(importance_file, index=False, encoding='utf-8-sig')
print(f"已保存: {importance_file}")

# 保存预测结果
results_file = f"{output_dir}/改进预处理_预测结果.csv"
results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
print(f"已保存: {results_file}")

# 保存特征列表
feature_info_file = f"{output_dir}/改进预处理_特征信息.txt"
with open(feature_info_file, 'w', encoding='utf-8') as f:
    f.write("改进数据预处理方法分析\n")
    f.write("="*60 + "\n\n")
    f.write("数值型特征:\n")
    for feat in numeric_features:
        f.write(f"  - {feat}\n")
    f.write(f"\n分类型特征:\n")
    for feat in categorical_features:
        f.write(f"  - {feat}\n")
    f.write(f"\n预处理方法:\n")
    f.write(f"  - 数值型特征: 中位数填充缺失值 + 标准化\n")
    f.write(f"  - 分类型特征: 众数填充缺失值 + 独热编码\n")
print(f"已保存: {feature_info_file}")

# 生成报告
report = f"""
改进数据预处理方法分析报告
{'='*60}

一、数据信息
- 原始数据形状: {data.shape}
- 选择的特征数量: {len(X.columns)}
- 样本量: {len(y)}
- 患病率: {y.mean():.2%}

二、特征分类
数值型特征 ({len(numeric_features)}):
"""

for feat in numeric_features:
    report += f"  - {feat}\n"

report += f"""
分类型特征 ({len(categorical_features)}):
"""

for feat in categorical_features:
    report += f"  - {feat}\n"

report += f"""
三、预处理方法
1. 数值型特征:
   - 使用中位数填充缺失值
   - 使用StandardScaler标准化

2. 分类型特征:
   - 使用众数填充缺失值
   - 使用OneHotEncoder独热编码

四、模型性能
测试集:
- 准确率: {test_accuracy:.4f}
- AUC: {test_auc:.4f}
- OOB得分: {model.named_steps['classifier'].oob_score_:.4f}

交叉验证:
- 准确率: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}
- AUC: {cv_auc_scores.mean():.4f} +/- {cv_auc_scores.std():.4f}

五、分类报告
"""

report += classification_report(y_test, y_test_pred, target_names=['未患病', '患病'])

report += f"""
六、特征重要性排序（Top 20）
"""

for i, row in feature_importance.head(20).iterrows():
    report += f"{i+1}. {row['特征']}: {row['重要性']:.4f}\n"

report += f"""
七、风险分层结果
"""

for level in ['低风险', '中风险', '高风险']:
    level_data = results_df[results_df['风险等级'] == level]
    if len(level_data) > 0:
        count = len(level_data)
        percentage = count / len(results_df) * 100
        actual_rate = level_data['实际值'].mean() * 100
        report += f"{level}: {count}人 ({percentage:.1f}%), 实际患病率: {actual_rate:.1f}%\n"

report += f"""
八、改进效果对比
改进前:
- AUC: 0.5213
- 特征重要性: 只有BMI和吸烟非零，其他都是0

改进后:
- AUC: {test_auc:.4f}
- 特征重要性: {'多个特征有非零值' if (feature_importance['重要性'] > 0).sum() > 2 else '仍然只有少数特征有非零值'}

九、结论
1. 使用了更合理的预处理方法
2. 数值型特征用中位数填充，分类型特征用独热编码
3. 模型性能: AUC={test_auc:.4f}
4. {'模型性能有所提升' if test_auc > 0.5213 else '模型性能仍然较低'}
5. {'多个特征对预测有贡献' if (feature_importance['重要性'] > 0).sum() > 2 else '特征重要性仍然集中在少数特征'}

十、论文应用建议
1. 在论文中详细描述数据预处理方法
2. 对比不同预处理方法的效果
3. 分析哪些特征对胆结石预测最重要
4. 基于风险分层结果，提出针对性的预防建议

十一、输出文件
1. 改进预处理_特征重要性.csv - 特征的重要性排序
2. 改进预处理_预测结果.csv - 预测结果和风险分层
3. 改进预处理_特征信息.txt - 特征分类和预处理方法
"""

report_file = f"{output_dir}/改进预处理分析报告.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"已保存: {report_file}")

print("\n" + "="*60)
print("改进数据预处理方法的分析完成！")
print("="*60)
print(f"模型性能: 准确率={test_accuracy:.4f}, AUC={test_auc:.4f}")
print(f"{'模型性能有所提升！' if test_auc > 0.5213 else '模型性能仍然较低'}")