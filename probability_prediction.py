"""
概率预测与风险分层分析
功能：使用清理后的数据，进行概率预测和风险分层
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

print("="*60)
print("概率预测与风险分层分析")
print("="*60)

output_dir = r"C:\Users\霍冠华\Documents\trae_projects\claude code"

# 1. 加载清理后的数据
print("\n1. 加载清理后的数据...")
data = pd.read_csv(f"{output_dir}/清理后数据.csv", encoding='utf-8-sig')
target_column = data.columns[-1]

print(f"数据形状: {data.shape}")
print(f"目标变量: {target_column}")

# 2. 准备数据
y = data[target_column]
X = data.drop(columns=[target_column])

# 3. 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")

# 4. 训练随机森林模型
print("\n2. 训练随机森林模型...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

rf_model.fit(X_train, y_train)
print("模型训练完成")

# 5. 概率校准（提高概率预测的准确性）
print("\n3. 概率校准...")
calibrated_model = CalibratedClassifierCV(rf_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)
print("概率校准完成")

# 6. 预测概率
print("\n4. 预测概率...")
# 训练集概率
train_prob = calibrated_model.predict_proba(X_train)[:, 1]
# 测试集概率
test_prob = calibrated_model.predict_proba(X_test)[:, 1]
# 所有数据概率
all_prob = calibrated_model.predict_proba(X)[:, 1]

print(f"训练集概率范围: {train_prob.min():.4f} - {train_prob.max():.4f}")
print(f"测试集概率范围: {test_prob.min():.4f} - {test_prob.max():.4f}")
print(f"所有数据概率范围: {all_prob.min():.4f} - {all_prob.max():.4f}")

# 7. 风险分层
print("\n5. 风险分层...")

# 定义风险分层阈值
risk_thresholds = {
    '低风险': (0.0, 0.33),
    '中风险': (0.33, 0.67),
    '高风险': (0.67, 1.0)
}

# 为所有样本分配风险等级
risk_levels = []
for prob in all_prob:
    if prob < 0.33:
        risk_levels.append('低风险')
    elif prob < 0.67:
        risk_levels.append('中风险')
    else:
        risk_levels.append('高风险')

# 创建包含概率和风险等级的数据框
results_df = pd.DataFrame({
    '实际值': y.values,
    '患病概率': all_prob,
    '风险等级': risk_levels
})

# 统计各风险等级的数量
risk_distribution = results_df['风险等级'].value_counts().sort_index()
print(f"\n风险等级分布:")
for level, count in risk_distribution.items():
    percentage = count / len(results_df) * 100
    print(f"  {level}: {count}人 ({percentage:.1f}%)")

# 统计各风险等级中的实际患病率
print(f"\n各风险等级中的实际患病率:")
for level in ['低风险', '中风险', '高风险']:
    level_data = results_df[results_df['风险等级'] == level]
    if len(level_data) > 0:
        actual_rate = level_data['实际值'].mean() * 100
        print(f"  {level}: {actual_rate:.1f}% (样本数: {len(level_data)})")

# 8. 保存结果
print("\n6. 保存结果...")

# 保存所有样本的预测结果
results_file = f"{output_dir}/概率预测与风险分层结果.csv"
results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
print(f"已保存预测结果: {results_file}")

# 保存测试集的预测结果
test_results_df = pd.DataFrame({
    '实际值': y_test.values,
    '患病概率': test_prob,
    '风险等级': ['低风险' if p < 0.33 else ('中风险' if p < 0.67 else '高风险') for p in test_prob]
})
test_results_file = f"{output_dir}/测试集概率预测结果.csv"
test_results_df.to_csv(test_results_file, index=False, encoding='utf-8-sig')
print(f"已保存测试集预测结果: {test_results_file}")

# 9. 可视化
print("\n7. 生成可视化图表...")

# 图表1: 概率分布直方图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1: 所有样本的概率分布
axes[0, 0].hist(all_prob, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(x=0.33, color='green', linestyle='--', linewidth=2, label='低风险阈值')
axes[0, 0].axvline(x=0.67, color='orange', linestyle='--', linewidth=2, label='高风险阈值')
axes[0, 0].set_xlabel('患病概率', fontsize=11)
axes[0, 0].set_ylabel('频数', fontsize=11)
axes[0, 0].set_title('所有样本的患病概率分布', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# 子图2: 实际患病 vs 未患病的概率分布对比
prob_0 = all_prob[y == 0]
prob_1 = all_prob[y == 1]
axes[0, 1].hist(prob_0, bins=30, alpha=0.5, label='未患病', color='lightgreen', edgecolor='darkgreen')
axes[0, 1].hist(prob_1, bins=30, alpha=0.5, label='患病', color='lightcoral', edgecolor='darkred')
axes[0, 1].set_xlabel('患病概率', fontsize=11)
axes[0, 1].set_ylabel('频数', fontsize=11)
axes[0, 1].set_title('实际患病 vs 未患病的概率分布', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# 子图3: 风险等级分布
risk_counts = results_df['风险等级'].value_counts()
colors = ['lightgreen', 'orange', 'red']
bars = axes[1, 0].bar(risk_counts.index, risk_counts.values, color=colors, edgecolor='black')
axes[1, 0].set_ylabel('人数', fontsize=11)
axes[1, 0].set_title('风险等级分布', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)

# 子图4: 各风险等级的实际患病率
risk_actual_rates = []
for level in ['低风险', '中风险', '高风险']:
    level_data = results_df[results_df['风险等级'] == level]
    if len(level_data) > 0:
        risk_actual_rates.append(level_data['实际值'].mean() * 100)
    else:
        risk_actual_rates.append(0)

bars = axes[1, 1].bar(['低风险', '中风险', '高风险'], risk_actual_rates, 
                       color=['lightgreen', 'orange', 'red'], edgecolor='black')
axes[1, 1].set_ylabel('实际患病率 (%)', fontsize=11)
axes[1, 1].set_title('各风险等级的实际患病率', fontsize=12, fontweight='bold')
axes[1, 1].set_ylim([0, 100])
axes[1, 1].grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f"{output_dir}/概率预测与风险分层分析图.png", dpi=300, bbox_inches='tight')
plt.close()
print("已保存: 概率预测与风险分层分析图.png")

# 图表2: ROC曲线（即使性能不好也要展示）
from sklearn.metrics import roc_curve, auc

fig, ax = plt.subplots(figsize=(8, 8))
fpr, tpr, thresholds = roc_curve(y_test, test_prob)
roc_auc = auc(fpr, tpr)

ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('假阳性率', fontsize=12)
ax.set_ylabel('真阳性率', fontsize=12)
ax.set_title('ROC曲线', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

plt.savefig(f"{output_dir}/ROC曲线_清理后数据.png", dpi=300, bbox_inches='tight')
plt.close()
print("已保存: ROC曲线_清理后数据.png")

# 10. 生成报告
report = f"""
概率预测与风险分层分析报告
{'='*60}

一、数据信息
- 数据形状: {data.shape}
- 目标变量: {target_column}
- 患病率: {y.mean():.2%}

二、模型信息
- 模型类型: 随机森林
- 树的数量: 100
- 最大深度: 8
- 概率校准: Isotonic回归

三、概率预测结果
- 训练集概率范围: {train_prob.min():.4f} - {train_prob.max():.4f}
- 测试集概率范围: {test_prob.min():.4f} - {test_prob.max():.4f}
- 所有数据概率范围: {all_prob.min():.4f} - {all_prob.max():.4f}

四、风险分层结果
风险分层标准:
- 低风险: 患病概率 < 33%
- 中风险: 33% ≤ 患病概率 < 67%
- 高风险: 患病概率 ≥ 67%

风险等级分布:
"""

for level, count in risk_distribution.items():
    percentage = count / len(results_df) * 100
    report += f"- {level}: {count}人 ({percentage:.1f}%)\n"

report += f"""
各风险等级的实际患病率:
"""

for level in ['低风险', '中风险', '高风险']:
    level_data = results_df[results_df['风险等级'] == level]
    if len(level_data) > 0:
        actual_rate = level_data['实际值'].mean() * 100
        report += f"- {level}: {actual_rate:.1f}% (样本数: {len(level_data)})\n"

report += f"""
五、结论
1. 使用概率预测可以更精细地评估个体的胆结石患病风险
2. 风险分层有助于识别高风险人群，进行重点干预
3. 当前模型的预测能力有限（AUC ≈ {roc_auc:.3f}），需要更多数据或更好的特征
4. 建议结合临床经验和专业知识，综合评估风险

六、输出文件
1. 概率预测与风险分层结果.csv - 所有样本的预测结果
2. 测试集概率预测结果.csv - 测试集的预测结果
3. 概率预测与风险分层分析图.png - 可视化图表
4. ROC曲线_清理后数据.png - ROC曲线
"""

report_file = f"{output_dir}/概率预测与风险分层分析报告.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"已保存报告: {report_file}")

print("\n" + "="*60)
print("概率预测与风险分层分析完成！")
print("="*60)
print("生成的文件:")
print("1. 概率预测与风险分层结果.csv")
print("2. 测试集概率预测结果.csv")
print("3. 概率预测与风险分层分析图.png")
print("4. ROC曲线_清理后数据.png")
print("5. 概率预测与风险分层分析报告.txt")