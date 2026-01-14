import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# 读取显著特征列表
with open(r'C:\Users\霍冠华\Documents\trae_projects\claude code\显著特征列表.txt', 'r', encoding='utf-8') as f:
    selected_features = [line.strip() for line in f.readlines()]

print(f"从文件中加载了 {len(selected_features)} 个显著特征")

# 加载数据
data = pd.read_csv(r'C:\Users\霍冠华\Documents\trae_projects\claude code\预处理数据.csv', encoding='utf-8-sig')
target_column = data.columns[-1]
print(f"目标变量: {target_column}")

# 筛选特征
columns_to_use = selected_features + [target_column]
data_filtered = data[columns_to_use]

# 分离特征和目标变量
y = data_filtered[target_column]
X = data_filtered.drop(columns=[target_column])

print(f"特征数量: {len(X.columns)}")

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 拟合Logistic回归模型
print("\n正在拟合Logistic回归模型...")
X_train_sm = sm.add_constant(X_train_scaled)
logit_model = sm.Logit(y_train, X_train_sm)

try:
    results = logit_model.fit(disp=False, maxiter=100)
    print("模型拟合成功")
except Exception as e:
    print(f"模型拟合失败: {e}")
    exit(1)

# 分析结果
print("\n正在分析结果...")
params = results.params
pvalues = results.pvalues
conf_int = results.conf_int()

results_df = pd.DataFrame({
    '特征': ['截距'] + X.columns.tolist(),
    '系数': params.values,
    '标准误': results.bse.values,
    'Z值': results.tvalues.values,
    'P值': pvalues.values,
    'OR值': np.exp(params.values),
    'OR_95%_CI_下限': np.exp(conf_int[0].values),
    'OR_95%_CI_上限': np.exp(conf_int[1].values)
})

# 添加显著性标记
results_df['显著性'] = results_df['P值'].apply(
    lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else ''))
)

# 筛选显著特征（不包括截距）
significant_features = results_df[
    (results_df['P值'] < 0.05) & (results_df['特征'] != '截距')
]

print(f"显著特征数量: {len(significant_features)}")

# 评估模型
print("\n正在评估模型性能...")
X_test_sm = sm.add_constant(X_test_scaled)
y_pred_prob = results.predict(X_test_sm)
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"准确率: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\n混淆矩阵:")
print(cm)

# 保存结果
output_dir = r"C:\Users\霍冠华\Documents\trae_projects\claude code"
results_file = f"{output_dir}/Logistic回归结果_筛选特征.csv"
results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
print(f"\n已保存Logistic回归结果: {results_file}")

if len(significant_features) > 0:
    significant_file = f"{output_dir}/Logistic回归_显著特征_筛选特征.csv"
    significant_features.to_csv(significant_file, index=False, encoding='utf-8-sig')
    print(f"已保存显著特征: {significant_file}")

# 生成分析报告
report = f"""
多因素Logistic回归分析报告（使用卡方检验筛选的特征）
{'='*60}

数据信息:
- 数据集形状: {data_filtered.shape}
- 目标变量: {target_column}
- 特征数量: {len(X.columns)}（从卡方检验筛选）
- 训练集大小: {X_train.shape[0]}
- 测试集大小: {X_test.shape[0]}

模型性能:
- 准确率: {accuracy:.4f}
- AUC: {auc:.4f}
"""

if len(significant_features) > 0:
    report += f"\n显著风险因素 (P < 0.05):\n"
    report += f"{'-'*60}\n"
    for i, row in significant_features.iterrows():
        report += f"{row['特征']}\n"
        report += f"  OR值: {row['OR值']:.4f} (95% CI: {row['OR_95%_CI_下限']:.4f}-{row['OR_95%_CI_上限']:.4f})\n"
        report += f"  P值: {row['P值']:.4f} {row['显著性']}\n\n"

report += f"""
结论:
通过多因素Logistic回归分析，在卡方检验筛选的 {len(X.columns)} 个特征中，
共识别出 {len(significant_features)} 个与胆结石发病显著相关的独立风险因素。
这些因素在控制其他变量后仍然显著，表明它们是胆结石发病的独立风险因素。

输出文件:
1. Logistic回归结果_筛选特征.csv - 筛选特征的回归结果
2. Logistic回归_显著特征_筛选特征.csv - 显著特征的回归结果
"""

report_file = f"{output_dir}/Logistic回归分析报告_筛选特征.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"已保存分析报告: {report_file}")

print("\n" + "="*60)
print("分析摘要")
print("="*60)
print(f"总特征数: {len(X.columns)}")
print(f"显著特征数: {len(significant_features)}")

if len(significant_features) > 0:
    print(f"\n显著风险因素:")
    for i, row in significant_features.iterrows():
        print(f"  {row['特征']}: OR={row['OR值']:.4f}, P={row['P值']:.4f}")

print("\n分析完成！")