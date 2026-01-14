"""
汉中市胆结石风险调研 - 随机森林特征重要性分析
功能：使用随机森林模型进行风险因素重要性排序
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# 加载数据
print("正在加载数据...")
data = pd.read_csv(r'C:\Users\霍冠华\Documents\trae_projects\claude code\预处理数据.csv', encoding='utf-8-sig')
target_column = data.columns[-1]

print(f"数据加载成功: {data.shape}")
print(f"目标变量: {target_column}")

# 分离特征和目标变量
y = data[target_column]
X = data.drop(columns=[target_column])

print(f"特征数量: {len(X.columns)}")

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")

# 创建并训练随机森林模型
print("\n正在训练随机森林模型...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

rf_model.fit(X_train, y_train)
print("随机森林模型训练完成")

# 获取特征重要性
feature_importance = rf_model.feature_importances_

# 创建特征重要性数据框
importance_df = pd.DataFrame({
    '特征': X.columns,
    '重要性': feature_importance
})

# 按重要性排序
importance_df = importance_df.sort_values('重要性', ascending=False)

# 筛选重要性大于0的特征
importance_df_filtered = importance_df[importance_df['重要性'] > 0]

print(f"\n重要性大于0的特征数量: {len(importance_df_filtered)}")

# 评估模型性能
print("\n正在评估模型性能...")
y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"准确率: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print(f"OOB得分: {rf_model.oob_score_:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(f"\n混淆矩阵:")
print(cm)

# 分类报告
print(f"\n分类报告:")
print(classification_report(y_test, y_pred))

# 保存结果
output_dir = r"C:\Users\霍冠华\Documents\trae_projects\claude code"

# 保存所有特征重要性
importance_file = f"{output_dir}/随机森林_特征重要性_所有特征.csv"
importance_df.to_csv(importance_file, index=False, encoding='utf-8-sig')
print(f"\n已保存所有特征重要性: {importance_file}")

# 保存重要性大于0的特征
importance_filtered_file = f"{output_dir}/随机森林_特征重要性_筛选特征.csv"
importance_df_filtered.to_csv(importance_filtered_file, index=False, encoding='utf-8-sig')
print(f"已保存筛选特征重要性: {importance_filtered_file}")

# 生成分析报告
report = f"""
随机森林特征重要性分析报告
{'='*60}

数据信息:
- 数据集形状: {data.shape}
- 目标变量: {target_column}
- 特征数量: {len(X.columns)}
- 训练集大小: {X_train.shape[0]}
- 测试集大小: {X_test.shape[0]}

模型参数:
- 树的数量: 100
- 最大深度: 10
- 最小分裂样本数: 5
- 最小叶子节点样本数: 2
- 最大特征数: sqrt

模型性能:
- 准确率: {accuracy:.4f}
- AUC: {auc:.4f}
- OOB得分: {rf_model.oob_score_:.4f}

特征重要性分析:
- 总特征数: {len(X.columns)}
- 重要性大于0的特征数: {len(importance_df_filtered)}

前20个最重要的特征:
{'-'*60}
"""

top_20_features = importance_df_filtered.head(20)
for i, row in top_20_features.iterrows():
    report += f"{row['特征']}: {row['重要性']:.4f}\n"

report += f"""
结论:
通过随机森林模型分析，共识别出 {len(importance_df_filtered)} 个对胆结石发病有影响的特征。
这些特征的重要性排序反映了它们在预测胆结石发病中的相对贡献度。

输出文件:
1. 随机森林_特征重要性_所有特征.csv - 所有特征的重要性排序
2. 随机森林_特征重要性_筛选特征.csv - 重要性大于0的特征排序
"""

report_file = f"{output_dir}/随机森林分析报告.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"已保存分析报告: {report_file}")

print("\n" + "="*60)
print("分析摘要")
print("="*60)
print(f"总特征数: {len(X.columns)}")
print(f"重要性大于0的特征数: {len(importance_df_filtered)}")

print(f"\n前10个最重要的特征:")
for i, row in importance_df_filtered.head(10).iterrows():
    print(f"  {i+1}. {row['特征']}: {row['重要性']:.4f}")

print("\n分析完成！")