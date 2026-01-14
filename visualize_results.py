"""
汉中市胆结石风险调研 - 结果可视化
功能：生成各种分析结果的可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# 设置图表风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

print("正在生成可视化图表...")

output_dir = r"C:\Users\霍冠华\Documents\trae_projects\claude code"

# 1. 读取数据
print("\n1. 读取分析结果...")
chi_square_results = pd.read_csv(
    f"{output_dir}/卡方检验结果_所有特征.csv",
    encoding='utf-8-sig'
)

rf_results = pd.read_csv(
    f"{output_dir}/随机森林_特征重要性_筛选特征.csv",
    encoding='utf-8-sig'
)

comprehensive_results = pd.read_csv(
    f"{output_dir}/综合特征重要性分析.csv",
    encoding='utf-8-sig'
)

# 2. 图表1: 卡方检验P值分布
print("\n2. 生成卡方检验P值分布图...")
fig, ax = plt.subplots(figsize=(12, 6))
p_values = chi_square_results['P值'].dropna()
ax.hist(p_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='显著性水平 (P=0.05)')
ax.set_xlabel('P值', fontsize=12)
ax.set_ylabel('频数', fontsize=12)
ax.set_title('卡方检验P值分布', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{output_dir}/卡方检验P值分布图.png", dpi=300, bbox_inches='tight')
plt.close()
print("已保存: 卡方检验P值分布图.png")

# 3. 图表2: 卡方检验显著特征P值
print("\n3. 生成卡方检验显著特征P值图...")
significant_chi = chi_square_results[chi_square_results['P值'] < 0.05].sort_values('P值')
if len(significant_chi) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(significant_chi)), significant_chi['P值'], color='coral')
    ax.set_yticks(range(len(significant_chi)))
    ax.set_yticklabels([f"{i+1}. {x[:30]}..." if len(x) > 30 else f"{i+1}. {x}" 
                       for i, x in enumerate(significant_chi['特征'])], fontsize=9)
    ax.set_xlabel('P值', fontsize=12)
    ax.set_title('卡方检验显著特征 (P < 0.05)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='P=0.05')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/卡方检验显著特征P值图.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: 卡方检验显著特征P值图.png")

# 4. 图表3: 随机森林特征重要性（Top 20）
print("\n4. 生成随机森林特征重要性图...")
top_20_rf = rf_results.head(20)
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(top_20_rf)), top_20_rf['重要性'], 
               color='lightcoral', edgecolor='darkred')
ax.set_yticks(range(len(top_20_rf)))
ax.set_yticklabels([f"{i+1}. {x[:35]}..." if len(x) > 35 else f"{i+1}. {x}" 
                   for i, x in enumerate(top_20_rf['特征'])], fontsize=9)
ax.set_xlabel('特征重要性', fontsize=12)
ax.set_title('随机森林特征重要性 (Top 20)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{output_dir}/随机森林特征重要性图.png", dpi=300, bbox_inches='tight')
plt.close()
print("已保存: 随机森林特征重要性图.png")

# 5. 图表4: 综合特征重要性（Top 20）
print("\n5. 生成综合特征重要性图...")
top_20_comprehensive = comprehensive_results.head(20)
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_20_comprehensive)))
bars = ax.barh(range(len(top_20_comprehensive)), top_20_comprehensive['综合评分'],
               color=colors, edgecolor='darkgreen')
ax.set_yticks(range(len(top_20_comprehensive)))
ax.set_yticklabels([f"{i+1}. {x[:35]}..." if len(x) > 35 else f"{i+1}. {x}" 
                   for i, x in enumerate(top_20_comprehensive['特征'])], fontsize=9)
ax.set_xlabel('综合评分', fontsize=12)
ax.set_title('综合特征重要性 (Top 20)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{output_dir}/综合特征重要性图.png", dpi=300, bbox_inches='tight')
plt.close()
print("已保存: 综合特征重要性图.png")

# 6. 图表5: 方法比较
print("\n6. 生成方法比较图...")
comparison_data = {
    '方法': ['卡方检验', 'Logistic回归', '随机森林'],
    '识别特征数': [len(chi_square_results[chi_square_results['P值'] < 0.05]), 0, len(rf_results)]
}
comparison_df = pd.DataFrame(comparison_data)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(comparison_df['方法'], comparison_df['识别特征数'],
               color=['steelblue', 'lightcoral', 'lightgreen'], edgecolor='black')
ax.set_ylabel('识别特征数', fontsize=12)
ax.set_title('三种方法识别特征数比较', fontsize=14, fontweight='bold')
ax.set_xlabel('方法', fontsize=12)

# 在柱子上添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(f"{output_dir}/方法比较图.png", dpi=300, bbox_inches='tight')
plt.close()
print("已保存: 方法比较图.png")

# 7. 图表6: 特征重叠Venn图（简化版）
print("\n7. 生成特征重叠图...")
chi_features = set(chi_square_results[chi_square_results['P值'] < 0.05]['特征'].tolist())
rf_features = set(rf_results['特征'].tolist())

fig, ax = plt.subplots(figsize=(10, 6))
overlap_data = {
    '卡方检验': len(chi_features),
    '随机森林': len(rf_features),
    '重叠': len(chi_features & rf_features)
}

x = [0, 1]
width = 0.35
bars1 = ax.bar([x[0]-width/2, x[1]-width/2], [overlap_data['卡方检验'], overlap_data['随机森林']],
               width, label='各方法独有', color=['steelblue', 'lightcoral'], edgecolor='black')
bars2 = ax.bar([x[0]+width/2, x[1]+width/2], [overlap_data['重叠'], overlap_data['重叠']],
               width, label='共同识别', color='lightgreen', edgecolor='black')

ax.set_ylabel('特征数', fontsize=12)
ax.set_title('卡方检验与随机森林特征重叠', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['卡方检验', '随机森林'], fontsize=11)
ax.legend(fontsize=10)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(f"{output_dir}/特征重叠图.png", dpi=300, bbox_inches='tight')
plt.close()
print("已保存: 特征重叠图.png")

# 8. 图表7: 综合评分分布
print("\n8. 生成综合评分分布图...")
fig, ax = plt.subplots(figsize=(12, 6))
scores = comprehensive_results['综合评分']
ax.hist(scores, bins=50, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
ax.axvline(x=scores.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'平均值: {scores.mean():.3f}')
ax.set_xlabel('综合评分', fontsize=12)
ax.set_ylabel('频数', fontsize=12)
ax.set_title('综合特征重要性评分分布', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{output_dir}/综合评分分布图.png", dpi=300, bbox_inches='tight')
plt.close()
print("已保存: 综合评分分布图.png")

# 9. 图表8: Top 10 特征综合对比
print("\n9. 生成Top 10特征综合对比图...")
top_10 = comprehensive_results.head(10)

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(top_10))
width = 0.25

# 准备数据
chi_p = top_10['卡方检验_P值'].fillna(1)
rf_imp = top_10['随机森林_重要性']
comp_score = top_10['综合评分']

# 标准化到0-1范围
chi_p_norm = 1 - chi_p  # P值越小，分数越高
chi_p_norm = chi_p_norm / chi_p_norm.max() if chi_p_norm.max() > 0 else chi_p_norm
rf_imp_norm = rf_imp / rf_imp.max() if rf_imp.max() > 0 else rf_imp
comp_score_norm = comp_score / comp_score.max() if comp_score.max() > 0 else comp_score

bars1 = ax.bar(x - width, chi_p_norm, width, label='卡方检验 (1-P值)', 
               color='steelblue', edgecolor='black')
bars2 = ax.bar(x, rf_imp_norm, width, label='随机森林重要性', 
               color='lightcoral', edgecolor='black')
bars3 = ax.bar(x + width, comp_score_norm, width, label='综合评分', 
               color='lightgreen', edgecolor='black')

ax.set_ylabel('标准化分数', fontsize=12)
ax.set_title('Top 10 特征在不同方法中的表现', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"{i+1}" for i in range(len(top_10))], fontsize=10)
ax.legend(fontsize=10)

# 添加特征名称注释
feature_labels = [f"{i+1}. {x[:20]}..." if len(x) > 20 else f"{i+1}. {x}" 
                  for i, x in enumerate(top_10['特征'])]
for i, label in enumerate(feature_labels):
    ax.annotate(label, xy=(i, 0), xytext=(0, -30),
                textcoords='offset points', ha='center', va='top',
                fontsize=8, rotation=45)

plt.tight_layout()
plt.savefig(f"{output_dir}/Top10特征综合对比图.png", dpi=300, bbox_inches='tight')
plt.close()
print("已保存: Top10特征综合对比图.png")

print("\n" + "="*60)
print("可视化图表生成完成！")
print("="*60)
print("生成的图表:")
print("1. 卡方检验P值分布图.png")
print("2. 卡方检验显著特征P值图.png")
print("3. 随机森林特征重要性图.png")
print("4. 综合特征重要性图.png")
print("5. 方法比较图.png")
print("6. 特征重叠图.png")
print("7. 综合评分分布图.png")
print("8. Top10特征综合对比图.png")
print(f"\n所有图表已保存到: {output_dir}")