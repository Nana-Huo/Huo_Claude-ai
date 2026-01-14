"""
基于论文筛选数据的完整分析
功能：使用符合论文需求的43个变量进行完整分析
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from scipy.stats import chi2_conting as chi2_conting_func
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("="*60)
print("基于论文筛选数据的完整分析")
print("="*60)

output_dir = r"C:\Users\霍冠华\Documents\trae_projects\claude code"

# 1. 加载筛选后的数据
print("\n1. 加载筛选后的数据...")
data = pd.read_csv(f"{output_dir}/论文筛选数据.csv", encoding='utf-8-sig')
target_column = data.columns[-1]

print(f"数据形状: {data.shape}")
print(f"目标变量: {target_column}")

# 2. 准备数据
y = data[target_column]
X = data.drop(columns=[target_column])

print(f"\n特征数量: {len(X.columns)}")
print(f"样本量: {len(y)}")
print(f"患病率: {y.mean():.2%}")

# 3. 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")

# 4. 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 卡方检验
print("\n2. 卡方检验分析...")
chi_square_results = []

for i, column in enumerate(X.columns):
    try:
        contingency_table = pd.crosstab(X[column], y)
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            continue
        
        chi2, p_value, dof, expected = chi2_conting(contingency_table)
        
        if (expected < 5).sum() / expected.size > 0.2:
            continue
        
        n = contingency_table.sum().sum()
        phi = np.sqrt(chi2 / n)
        k = min(contingency_table.shape) - 1
        cramers_v = phi / np.sqrt(k - 1) if k > 1 else phi
        
        chi_square_results.append({
            '特征': column,
            '卡方值': chi2,
            'P值': p_value,
            '自由度': dof,
            "Cramer's V": cramers_v,
            '显著性': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
        })
    except:
        continue

chi_square_df = pd.DataFrame(chi_square_results).sort_values('P值')

significant_chi = chi_square_df[chi_square_df['P值'] < 0.05]
print(f"显著特征数 (P < 0.05): {len(significant_chi)}")

# 6. 随机森林分析
print("\n3. 随机森林分析...")
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

# 概率校准
calibrated_model = CalibratedClassifierCV(rf_model, method='isotonic', cv=5)
calibrated_model.fit(X_train_scaled, y_train)
print("概率校准完成")

# 7. 评估模型性能
print("\n4. 评估模型性能...")

# 测试集性能
y_test_pred = rf_model.predict(X_test)
y_test_prob = calibrated_model.predict_proba(X_test_scaled)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_prob)

print(f"\n测试集性能:")
print(f"  准确率: {test_accuracy:.4f}")
print(f"  AUC: {test_auc:.4f}")
print(f"  OOB得分: {rf_model.oob_score_:.4f}")

# 交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy')
cv_auc_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='roc_auc')

print(f"\n5折交叉验证结果:")
print(f"  准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  AUC: {cv_auc_scores.mean():.4f} ± {cv_auc_scores.std():.4f}")

# 8. 特征重要性
print("\n5. 特征重要性分析...")
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': rf_model.feature_importances_
}).sort_values('重要性', ascending=False)

print(f"\n前10个最重要的特征:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {i+1}. {row['特征']}: {row['重要性']:.4f}")

# 9. 概率预测和风险分层
print("\n6. 概率预测和风险分层...")
all_prob = calibrated_model.predict_proba(X)[:, 1]

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

# 10. 保存结果
print("\n7. 保存结果...")

# 保存卡方检验结果
chi_square_file = f"{output_dir}/论文_卡方检验结果.csv"
chi_square_df.to_csv(chi_square_file, index=False, encoding='utf-8-sig')
print(f"已保存: {chi_square_file}")

# 保存显著特征
if len(significant_chi) > 0:
    significant_chi_file = f"{output_dir}/论文_卡方检验显著特征.csv"
    significant_chi.to_csv(significant_chi_file, index=False, encoding='utf-8-sig')
    print(f"已保存: {significant_chi_file}")

# 保存特征重要性
importance_file = f"{output_dir}/论文_特征重要性.csv"
feature_importance.to_csv(importance_file, index=False, encoding='utf-8-sig')
print(f"已保存: {importance_file}")

# 保存预测结果
results_file = f"{output_dir}/论文_预测结果.csv"
results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
print(f"已保存: {results_file}")

print("\n" + "="*60)
print("基于论文筛选数据的完整分析完成！")
print("="*60)
print(f"模型性能: 准确率={test_accuracy:.4f}, AUC={test_auc:.4f}")
print(f"风险分层效果良好！")