"""
诊断数据泄露和过拟合问题
功能：检查是否存在数据泄露，分析模型性能异常的原因
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

print("="*60)
print("数据泄露和过拟合诊断")
print("="*60)

# 加载数据
print("\n1. 加载数据...")
data = pd.read_csv(r'C:\Users\霍冠华\Documents\trae_projects\claude code\预处理数据.csv', encoding='utf-8-sig')
target_column = data.columns[-1]

print(f"数据形状: {data.shape}")
print(f"目标变量: {target_column}")

# 检查目标变量分布
y = data[target_column]
print(f"\n目标变量分布:")
print(y.value_counts())
print(f"患病率: {y.mean():.2%}")

# 2. 检查是否有与目标变量完全相关的特征
print("\n2. 检查特征与目标变量的相关性...")
X = data.drop(columns=[target_column])
feature_correlations = []

for col in X.columns:
    correlation = abs(X[col].corr(y))
    if correlation > 0.8:  # 高度相关
        feature_correlations.append({
            '特征': col,
            '相关性': correlation,
            '唯一值数': X[col].nunique()
        })

if len(feature_correlations) > 0:
    print(f"\n发现 {len(feature_correlations)} 个高度相关的特征（相关性 > 0.8）:")
    for feat in feature_correlations:
        print(f"  {feat['特征']}: 相关性={feat['相关性']:.4f}, 唯一值数={feat['唯一值数']}")
else:
    print("\n未发现高度相关的特征")

# 3. 检查"了解胆结石的方式"特征
print("\n3. 检查潜在泄露特征...")
if '3.您之前通过什么方式了解到胆结石的？' in X.columns:
    feature_name = '3.您之前通过什么方式了解到胆结石的？'
    cross_tab = pd.crosstab(X[feature_name], y)
    print(f"\n'{feature_name}' 与目标变量的交叉表:")
    print(cross_tab)
    
    # 检查是否完全相关
    for val in cross_tab.index:
        if cross_tab.loc[val, 1] > 0 and cross_tab.loc[val, 0] == 0:
            print(f"  警告: 值 '{val}' 只出现在患病组中！")
        elif cross_tab.loc[val, 0] > 0 and cross_tab.loc[val, 1] == 0:
            print(f"  警告: 值 '{val}' 只出现在健康组中！")

# 4. 使用交叉验证重新评估模型
print("\n4. 使用交叉验证重新评估模型...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# 5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy')
cv_auc_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='roc_auc')

print(f"\n5折交叉验证结果:")
print(f"  准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  AUC: {cv_auc_scores.mean():.4f} ± {cv_auc_scores.std():.4f}")
print(f"  各折准确率: {cv_scores}")
print(f"  各折AUC: {cv_auc_scores}")

# 5. 检查训练集和测试集的性能差异
print("\n5. 检查训练集和测试集性能差异...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

rf_model.fit(X_train, y_train)
train_pred = rf_model.predict(X_train)
train_prob = rf_model.predict_proba(X_train)[:, 1]
test_pred = rf_model.predict(X_test)
test_prob = rf_model.predict_proba(X_test)[:, 1]

train_accuracy = accuracy_score(y_train, train_pred)
train_auc = roc_auc_score(y_train, train_prob)
test_accuracy = accuracy_score(y_test, test_pred)
test_auc = roc_auc_score(y_test, test_prob)

print(f"\n训练集性能:")
print(f"  准确率: {train_accuracy:.4f}")
print(f"  AUC: {train_auc:.4f}")

print(f"\n测试集性能:")
print(f"  准确率: {test_accuracy:.4f}")
print(f"  AUC: {test_auc:.4f}")

print(f"\n性能差异:")
print(f"  准确率差异: {train_accuracy - test_accuracy:.4f}")
print(f"  AUC差异: {train_auc - test_auc:.4f}")

if train_accuracy - test_accuracy > 0.1 or train_auc - test_auc > 0.1:
    print("\n警告: 训练集和测试集性能差异较大，可能存在过拟合！")

# 6. 检查特征重要性
print("\n6. 检查特征重要性...")
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': rf_model.feature_importances_
}).sort_values('重要性', ascending=False)

print(f"\n前10个最重要的特征:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {i+1}. {row['特征']}: {row['重要性']:.4f}")

# 7. 检查是否有"完美预测"的特征
print("\n7. 检查是否有完美预测的特征...")
perfect_predictors = []
for col in X.columns:
    # 检查该特征是否可以完美预测目标变量
    unique_values = X[col].unique()
    is_perfect = True
    
    for val in unique_values:
        mask = X[col] == val
        if mask.sum() > 0:
            unique_targets = y[mask].unique()
            if len(unique_targets) > 1:
                is_perfect = False
                break
    
    if is_perfect and X[col].nunique() > 1:
        perfect_predictors.append(col)

if len(perfect_predictors) > 0:
    print(f"\n发现 {len(perfect_predictors)} 个可以完美预测的特征:")
    for feat in perfect_predictors:
        print(f"  {feat}")
else:
    print("\n未发现可以完美预测的特征")

# 8. 诊断结论
print("\n" + "="*60)
print("诊断结论")
print("="*60)

issues_found = []

# 检查1: 高度相关特征
if len(feature_correlations) > 0:
    issues_found.append("发现高度相关的特征，可能存在数据泄露")

# 检查2: 交叉验证性能不稳定
if cv_scores.std() > 0.1 or cv_auc_scores.std() > 0.1:
    issues_found.append("交叉验证性能不稳定，模型泛化能力差")

# 检查3: 训练集和测试集性能差异大
if train_accuracy - test_accuracy > 0.1 or train_auc - test_auc > 0.1:
    issues_found.append("训练集和测试集性能差异大，存在过拟合")

# 检查4: 完美预测特征
if len(perfect_predictors) > 0:
    issues_found.append(f"发现 {len(perfect_predictors)} 个可以完美预测的特征，存在数据泄露")

# 检查5: AUC=1.0
if test_auc >= 0.99:
    issues_found.append("测试集AUC接近1.0，模型过于完美，可能存在问题")

if len(issues_found) > 0:
    print("\n发现以下问题:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
    
    print("\n建议:")
    print("  1. 移除可能泄露数据的特征（如'了解胆结石的方式'）")
    print("  2. 使用更严格的交叉验证")
    print("  3. 减少模型复杂度，防止过拟合")
    print("  4. 增加正则化")
    print("  5. 使用更保守的评估方法")
else:
    print("\n未发现明显的数据泄露或过拟合问题")

# 保存诊断报告
report = f"""
数据泄露和过拟合诊断报告
{'='*60}

一、数据基本信息
- 数据形状: {data.shape}
- 目标变量: {target_column}
- 患病率: {y.mean():.2%}

二、特征相关性分析
发现 {len(feature_correlations)} 个高度相关的特征（相关性 > 0.8）

三、交叉验证结果
- 准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
- AUC: {cv_auc_scores.mean():.4f} ± {cv_auc_scores.std():.4f}

四、训练集和测试集性能对比
- 训练集准确率: {train_accuracy:.4f}
- 测试集准确率: {test_accuracy:.4f}
- 准确率差异: {train_accuracy - test_accuracy:.4f}

- 训练集AUC: {train_auc:.4f}
- 测试集AUC: {test_auc:.4f}
- AUC差异: {train_auc - test_auc:.4f}

五、发现的问题
"""

if len(issues_found) > 0:
    for i, issue in enumerate(issues_found, 1):
        report += f"{i}. {issue}\n"
else:
    report += "未发现明显的数据泄露或过拟合问题\n"

report += f"""
六、建议
1. 移除可能泄露数据的特征（如'了解胆结石的方式'）
2. 使用更严格的交叉验证
3. 减少模型复杂度，防止过拟合
4. 增加正则化
5. 使用更保守的评估方法
"""

output_dir = r"C:\Users\霍冠华\Documents\trae_projects\claude code"
report_file = f"{output_dir}/诊断报告.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n诊断报告已保存: {report_file}")