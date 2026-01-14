"""
汉中市胆结石风险调研 - 方法比较与综合分析
功能：比较卡方检验、Logistic回归、随机森林三种方法的分析结果
"""

import pandas as pd
import numpy as np

print("="*60)
print("汉中市胆结石风险调研 - 方法比较与综合分析")
print("="*60)

# 1. 读取卡方检验结果
print("\n1. 读取卡方检验结果...")
chi_square_results = pd.read_csv(
    r'C:\Users\霍冠华\Documents\trae_projects\claude code\卡方检验结果_显著特征.csv',
    encoding='utf-8-sig'
)
print(f"卡方检验显著特征数: {len(chi_square_results)}")

# 2. 读取Logistic回归结果
print("\n2. 读取Logistic回归结果...")
try:
    logistic_results = pd.read_csv(
        r'C:\Users\霍冠华\Documents\trae_projects\claude code\Logistic回归结果_筛选特征.csv',
        encoding='utf-8-sig'
    )
    # 筛选显著特征（不包括截距）
    logistic_significant = logistic_results[
        (logistic_results['P值'] < 0.05) & (logistic_results['特征'] != '截距')
    ]
    print(f"Logistic回归显著特征数: {len(logistic_significant)}")
except:
    logistic_significant = pd.DataFrame(columns=['特征', 'P值', 'OR值'])
    print("Logistic回归无显著特征")

# 3. 读取随机森林结果
print("\n3. 读取随机森林结果...")
rf_results = pd.read_csv(
    r'C:\Users\霍冠华\Documents\trae_projects\claude code\随机森林_特征重要性_筛选特征.csv',
    encoding='utf-8-sig'
)
print(f"随机森林重要性>0的特征数: {len(rf_results)}")

# 4. 比较三种方法的结果
print("\n4. 比较三种方法的结果...")

# 提取各方法的特征列表
chi_square_features = set(chi_square_results['特征'].tolist())
logistic_features = set(logistic_significant['特征'].tolist())
rf_features = set(rf_results['特征'].tolist())

# 计算交集和并集
chi_square_logistic_intersection = chi_square_features & logistic_features
chi_square_rf_intersection = chi_square_features & rf_features
logistic_rf_intersection = logistic_features & rf_features
all_methods_intersection = chi_square_features & logistic_features & rf_features

all_methods_union = chi_square_features | logistic_features | rf_features

print(f"\n特征重叠分析:")
print(f"  卡方检验 & Logistic回归: {len(chi_square_logistic_intersection)} 个特征")
print(f"  卡方检验 & 随机森林: {len(chi_square_rf_intersection)} 个特征")
print(f"  Logistic回归 & 随机森林: {len(logistic_rf_intersection)} 个特征")
print(f"  三种方法共同识别: {len(all_methods_intersection)} 个特征")
print(f"  任意方法识别: {len(all_methods_union)} 个特征")

# 5. 创建综合特征重要性表
print("\n5. 创建综合特征重要性表...")

# 为所有特征创建综合评分
comprehensive_importance = []

for feature in all_methods_union:
    chi_square_p = chi_square_results[chi_square_results['特征'] == feature]['P值']
    chi_square_p = chi_square_p.values[0] if len(chi_square_p) > 0 else np.nan
    
    logistic_or = logistic_significant[logistic_significant['特征'] == feature]['OR值']
    logistic_or = logistic_or.values[0] if len(logistic_or) > 0 else np.nan
    
    rf_importance = rf_results[rf_results['特征'] == feature]['重要性']
    rf_importance = rf_importance.values[0] if len(rf_importance) > 0 else 0
    
    # 计算综合评分（归一化后加权平均）
    score = 0
    if not np.isnan(chi_square_p):
        score += (1 - chi_square_p)  # P值越小，分数越高
    
    if not np.isnan(logistic_or):
        if logistic_or > 1:
            score += min((logistic_or - 1) / 2, 1)  # OR值>1，分数增加
        else:
            score += min((1 - logistic_or) / 2, 1)  # OR值<1，分数减少
    
    score += rf_importance * 2  # 随机森林重要性权重较高
    
    comprehensive_importance.append({
        '特征': feature,
        '卡方检验_P值': chi_square_p,
        'Logistic回归_OR值': logistic_or,
        '随机森林_重要性': rf_importance,
        '综合评分': score
    })

# 创建数据框并排序
comprehensive_df = pd.DataFrame(comprehensive_importance)
comprehensive_df = comprehensive_df.sort_values('综合评分', ascending=False)

# 6. 保存综合结果
output_dir = r"C:\Users\霍冠华\Documents\trae_projects\claude code"

# 保存综合特征重要性
comprehensive_file = f"{output_dir}/综合特征重要性分析.csv"
comprehensive_df.to_csv(comprehensive_file, index=False, encoding='utf-8-sig')
print(f"已保存综合特征重要性: {comprehensive_file}")

# 保存方法比较结果
comparison_data = {
    '方法': ['卡方检验', 'Logistic回归', '随机森林'],
    '识别特征数': [len(chi_square_features), len(logistic_features), len(rf_features)],
    '特征数': [len(chi_square_results), len(logistic_significant), len(rf_results)]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_file = f"{output_dir}/方法比较结果.csv"
comparison_df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
print(f"已保存方法比较结果: {comparison_file}")

# 7. 生成综合分析报告
report = f"""
汉中市胆结石风险调研 - 方法比较与综合分析报告
{'='*60}

一、方法概述
{'-'*60}

1. 卡方检验
   - 功能：筛选与胆结石发病显著相关的潜在风险因素
   - 显著性水平：P < 0.05
   - 识别特征数：{len(chi_square_results)}

2. Logistic回归
   - 功能：识别独立风险因素，计算OR值和置信区间
   - 显著性水平：P < 0.05
   - 识别特征数：{len(logistic_significant)}

3. 随机森林
   - 功能：进行特征重要性排序
   - 重要性阈值：> 0
   - 识别特征数：{len(rf_results)}

二、方法比较
{'-'*60}

特征重叠分析：
- 卡方检验 & Logistic回归: {len(chi_square_logistic_intersection)} 个特征
- 卡方检验 & 随机森林: {len(chi_square_rf_intersection)} 个特征
- Logistic回归 & 随机森林: {len(logistic_rf_intersection)} 个特征
- 三种方法共同识别: {len(all_methods_intersection)} 个特征
- 任意方法识别: {len(all_methods_union)} 个特征

"""

if len(all_methods_intersection) > 0:
    report += f"三种方法共同识别的特征:\n"
    for feature in all_methods_intersection:
        report += f"  - {feature}\n"
else:
    report += f"三种方法未共同识别出显著特征\n"

report += f"""

三、综合分析
{'-'*60}

基于三种方法的分析结果，我们采用综合评分的方法对特征进行排序。
综合评分考虑了卡方检验的P值、Logistic回归的OR值和随机森林的重要性。

前20个最重要的风险因素:
{'-'*60}
"""

top_20_features = comprehensive_df.head(20)
for i, row in top_20_features.iterrows():
    report += f"{i+1}. {row['特征']}\n"
    report += f"   卡方检验P值: {row['卡方检验_P值']:.4f}\n" if not np.isnan(row['卡方检验_P值']) else "   卡方检验P值: N/A\n"
    report += f"   Logistic回归OR值: {row['Logistic回归_OR值']:.4f}\n" if not np.isnan(row['Logistic回归_OR值']) else "   Logistic回归OR值: N/A\n"
    report += f"   随机森林重要性: {row['随机森林_重要性']:.4f}\n"
    report += f"   综合评分: {row['综合评分']:.4f}\n\n"

report += f"""

四、结论
{'-'*60}

1. 方法一致性：
   - 三种方法共同识别出 {len(all_methods_intersection)} 个显著风险因素
   - 这 {len(all_methods_intersection)} 个特征在不同方法中都表现出与胆结石发病的显著关联
   - 这些特征具有较高的可信度，应作为重点关注的胆结石风险因素

2. 方法差异性：
   - 卡方检验主要识别单因素关联，未考虑变量间的相互作用
   - Logistic回归识别独立风险因素，考虑了其他变量的控制作用
   - 随机森林考虑了非线性关系和变量间的相互作用

3. 综合建议：
   - 优先关注综合评分前10位的特征
   - 特别是三种方法共同识别的特征
   - 结合汉中地区的饮食文化、生活习惯特点进行针对性分析

输出文件:
1. 综合特征重要性分析.csv - 所有特征的综合评分排序
2. 方法比较结果.csv - 三种方法的比较结果
"""

report_file = f"{output_dir}/方法比较与综合分析报告.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"已保存综合分析报告: {report_file}")

print("\n" + "="*60)
print("分析摘要")
print("="*60)
print(f"总识别特征数: {len(all_methods_union)}")
print(f"三种方法共同识别: {len(all_methods_intersection)}")

print(f"\n综合评分前10位的特征:")
for i, row in comprehensive_df.head(10).iterrows():
    print(f"  {i+1}. {row['特征']}: {row['综合评分']:.4f}")

print("\n分析完成！")