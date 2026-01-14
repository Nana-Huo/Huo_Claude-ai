"""
风险因素影响分析
功能：分析不同风险因素对患病概率的影响
"""

import pandas as pd
import numpy as np
from scipy import stats
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
print("风险因素影响分析")
print("="*60)

output_dir = r"C:\Users\霍冠华\Documents\trae_projects\claude code"

# 1. 加载数据
print("\n1. 加载数据...")
data = pd.read_csv(f"{output_dir}/清理后数据.csv", encoding='utf-8-sig')
probability_results = pd.read_csv(f"{output_dir}/概率预测与风险分层结果.csv", encoding='utf-8-sig')

target_column = data.columns[-1]

# 合并概率预测结果
data_with_prob = data.copy()
data_with_prob['患病概率'] = probability_results['患病概率'].values
data_with_prob['风险等级'] = probability_results['风险等级'].values

print(f"数据形状: {data_with_prob.shape}")

# 2. 分析数值型特征对患病概率的影响
print("\n2. 分析数值型特征对患病概率的影响...")

# 识别数值型特征
numeric_features = data_with_prob.select_dtypes(include=[np.number]).columns
# 排除目标变量和概率列
numeric_features = [col for col in numeric_features if col not in [target_column, '患病概率']]

print(f"数值型特征数量: {len(numeric_features)}")

# 计算相关系数
numeric_correlations = []
for feature in numeric_features:
    if data_with_prob[feature].nunique() > 1:  # 确保有多个唯一值
        corr, p_value = stats.pearsonr(data_with_prob[feature], data_with_prob['患病概率'])
        if not np.isnan(corr):
            numeric_correlations.append({
                '特征': feature,
                '相关系数': corr,
                'P值': p_value,
                '显著性': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else ''))
            })

numeric_corr_df = pd.DataFrame(numeric_correlations).sort_values('相关系数', key=abs, ascending=False)

print(f"\n相关系数绝对值 > 0.1 的数值型特征:")
significant_numeric = numeric_corr_df[abs(numeric_corr_df['相关系数']) > 0.1]
for i, row in significant_numeric.head(10).iterrows():
    print(f"  {row['特征']}: r={row['相关系数']:.4f}, p={row['P值']:.4f} {row['显著性']}")

# 3. 分析分类型特征对患病概率的影响
print("\n3. 分析分类型特征对患病概率的影响...")

# 识别分类型特征
categorical_features = data_with_prob.select_dtypes(include=['object']).columns
# 排除风险等级列
categorical_features = [col for col in categorical_features if col != '风险等级']

print(f"分类型特征数量: {len(categorical_features)}")

# 计算各分类特征不同类别的平均患病概率
categorical_effects = []
for feature in categorical_features:
    if data_with_prob[feature].nunique() <= 10 and data_with_prob[feature].nunique() > 1:  # 限制类别数量
        try:
            group_means = data_with_prob.groupby(feature)['患病概率'].mean()
            group_counts = data_with_prob.groupby(feature).size()
            
            # 计算类别间的差异
            if len(group_means) > 1:
                max_prob = group_means.max()
                min_prob = group_means.min()
                diff = max_prob - min_prob
                
                # ANOVA检验
                groups = [data_with_prob[data_with_prob[feature] == val]['患病概率'].values 
                         for val in group_means.index if val in data_with_prob[feature].values]
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                except:
                    f_stat, p_value = np.nan, np.nan
                
                categorical_effects.append({
                    'feature': feature,
                    'category_count': len(group_means),
                    'max_prob': max_prob,
                    'min_prob': min_prob,
                    'prob_diff': diff,
                    'f_stat': f_stat,
                    'p_value': p_value
                })
        except Exception as e:
            print(f"处理特征 {feature} 时出错: {e}")
            continue

if len(categorical_effects) > 0:
    categorical_effects_df = pd.DataFrame(categorical_effects).sort_values('prob_diff', ascending=False)
else:
    categorical_effects_df = pd.DataFrame(columns=['feature', 'category_count', 'max_prob', 'min_prob', 'prob_diff', 'f_stat', 'p_value'])
    print("未找到符合条件的分类型特征")

print(f"\n概率差 > 0.1 的分类型特征:")
significant_categorical = categorical_effects_df[categorical_effects_df['prob_diff'] > 0.1]
for i, row in significant_categorical.head(10).iterrows():
    print(f"  {row['feature']}: 概率差={row['prob_diff']:.4f}, F={row['f_stat']:.4f}, p={row['p_value']:.4f}")

# 4. 分析风险等级与特征的关系
print("\n4. 分析风险等级与特征的关系...")

risk_feature_analysis = []

# 分析数值型特征在不同风险等级的分布
for feature in significant_numeric['特征'].head(10):
    low_risk = data_with_prob[data_with_prob['风险等级'] == '低风险'][feature].mean()
    medium_risk = data_with_prob[data_with_prob['风险等级'] == '中风险'][feature].mean()
    high_risk = data_with_prob[data_with_prob['风险等级'] == '高风险'][feature].mean()
    
    risk_feature_analysis.append({
                    'feature': feature,
                    'low_risk_mean': low_risk,
                    'medium_risk_mean': medium_risk,
                    'high_risk_mean': high_risk,
                    'trend': 'increasing' if high_risk > low_risk else 'decreasing'
                })
    
    risk_feature_df = pd.DataFrame(risk_feature_analysis)
    
    print(f"\n数值型特征在不同风险等级的均值:")
    for i, row in risk_feature_df.head(5).iterrows():
        print(f"  {row['feature']}: 低={row['low_risk_mean']:.2f}, 中={row['medium_risk_mean']:.2f}, 高={row['high_risk_mean']:.2f}, 趋势={row['trend']}")
# 5. 可视化
print("\n5. 生成可视化图表...")

# 图表1: 数值型特征相关性热力图
if len(significant_numeric) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 选择前15个最相关的特征
    top_features = significant_numeric.head(15)
    correlation_matrix = data_with_prob[top_features['特征'].tolist() + ['患病概率']].corr()
    
    # 只显示与患病概率的相关性
    prob_correlations = correlation_matrix['患病概率'].drop('患病概率').sort_values(key=abs, ascending=False)
    
    colors = ['red' if x > 0 else 'blue' for x in prob_correlations.values]
    bars = ax.barh(range(len(prob_correlations)), prob_correlations.values, color=colors, edgecolor='black')
    ax.set_yticks(range(len(prob_correlations)))
    ax.set_yticklabels([f"{i+1}. {x[:35]}..." if len(x) > 35 else f"{i+1}. {x}" 
                       for i, x in enumerate(prob_correlations.index)], fontsize=9)
    ax.set_xlabel('与患病概率的相关系数', fontsize=12)
    ax.set_title(f'数值型特征与患病概率的相关性 (Top 15)', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/数值型特征相关性图.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: 数值型特征相关性图.png")

# 图表2: 分类型特征的概率差异
if len(significant_categorical) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_categorical = significant_categorical.head(10)
    bars = ax.barh(range(len(top_categorical)), top_categorical['prob_diff'], 
                   color='coral', edgecolor='darkred')
    ax.set_yticks(range(len(top_categorical)))
    ax.set_yticklabels([f"{i+1}. {x[:35]}..." if len(x) > 35 else f"{i+1}. {x}" 
                       for i, x in enumerate(top_categorical['feature'])], fontsize=9)
    ax.set_xlabel('概率差 (最大概率 - 最小概率)', fontsize=12)
    ax.set_title(f'分类型特征对患病概率的影响 (Top 10)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/分类型特征概率差异图.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: 分类型特征概率差异图.png")

# 图表3: 风险等级与关键特征的关系
if len(risk_feature_df) > 0:
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(risk_feature_df.head(6)))
    width = 0.25
    
    low_values = risk_feature_df.head(6)['low_risk_mean'].values
    medium_values = risk_feature_df.head(6)['medium_risk_mean'].values
    high_values = risk_feature_df.head(6)['high_risk_mean'].values
    
    bars1 = ax.bar(x - width, low_values, width, label='低风险', 
                   color='lightgreen', edgecolor='darkgreen')
    bars2 = ax.bar(x, medium_values, width, label='中风险', 
                   color='orange', edgecolor='darkorange')
    bars3 = ax.bar(x + width, high_values, width, label='高风险', 
                   color='red', edgecolor='darkred')
    
    ax.set_ylabel('特征均值', fontsize=12)
    ax.set_title('关键特征在不同风险等级的分布', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i+1}" for i in range(len(risk_feature_df.head(6)))], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加特征名称
    feature_labels = [f"{i+1}. {x[:20]}..." if len(x) > 20 else f"{i+1}. {x}" 
                      for i, x in enumerate(risk_feature_df.head(6)['feature'])]
    for i, label in enumerate(feature_labels):
        ax.annotate(label, xy=(i, 0), xytext=(0, -30),
                   textcoords='offset points', ha='center', va='top',
                   fontsize=8, rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/风险等级与特征关系图.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: 风险等级与特征关系图.png")

# 6. 保存结果
print("\n6. 保存结果...")

# 保存数值型特征相关性
if len(numeric_corr_df) > 0:
    numeric_corr_file = f"{output_dir}/数值型特征相关性分析.csv"
    numeric_corr_df.to_csv(numeric_corr_file, index=False, encoding='utf-8-sig')
    print(f"已保存: {numeric_corr_file}")

# 保存分类型特征影响
if len(categorical_effects_df) > 0:
    categorical_effects_file = f"{output_dir}/分类型特征影响分析.csv"
    categorical_effects_df.to_csv(categorical_effects_file, index=False, encoding='utf-8-sig')
    print(f"已保存: {categorical_effects_file}")

# 保存风险等级特征分析
if len(risk_feature_df) > 0:
    risk_feature_file = f"{output_dir}/风险等级特征分析.csv"
    risk_feature_df.to_csv(risk_feature_file, index=False, encoding='utf-8-sig')
    print(f"已保存: {risk_feature_file}")

# 7. 生成报告
report = f"""
风险因素影响分析报告
{'='*60}

一、数据信息
- 数据形状: {data_with_prob.shape}
- 目标变量: {target_column}
- 患病率: {data_with_prob[target_column].mean():.2%}

二、数值型特征分析
识别出 {len(numeric_features)} 个数值型特征

与患病概率相关性最强的特征 (Top 10):
"""

if len(significant_numeric) > 0:
    for i, row in significant_numeric.head(10).iterrows():
        report += f"{i+1}. {row['特征']}\n"
        report += f"   相关系数: {row['相关系数']:.4f}\n"
        report += f"   P值: {row['P值']:.4f} {row['显著性']}\n\n"
else:
    report += "未发现显著相关的数值型特征\n"

report += f"""
三、分类型特征分析
识别出 {len(categorical_features)} 个分类型特征

对患病概率影响最大的特征 (Top 10):
"""

if len(significant_categorical) > 0:
    for i, row in significant_categorical.head(10).iterrows():
        report += f"{i+1}. {row['特征']}\n"
        report += f"   类别数: {row['类别数']}\n"
        report += f"   概率差: {row['概率差']:.4f}\n"
        report += f"   F统计量: {row['F统计量']:.4f}\n"
        report += f"   P值: {row['P值']:.4f}\n\n"
else:
    report += "未发现显著影响的分类型特征\n"

report += f"""
四、风险等级与特征关系
"""

if len(risk_feature_df) > 0:
    report += f"关键特征在不同风险等级的均值:\n"
    for i, row in risk_feature_df.head(5).iterrows():
        report += f"{i+1}. {row['特征']}\n"
        report += f"   低风险: {row['低风险均值']:.2f}\n"
        report += f"   中风险: {row['中风险均值']:.2f}\n"
        report += f"   高风险: {row['高风险均值']:.2f}\n"
        report += f"   趋势: {row['趋势']}\n\n"
else:
    report += "未找到风险等级与特征的显著关系\n"

report += f"""
五、结论
1. 数值型特征中，{len(significant_numeric)} 个特征与患病概率显著相关
2. 分类型特征中，{len(significant_categorical)} 个特征对患病概率有显著影响
3. 风险等级与部分特征存在关联，可用于风险分层
4. 建议重点关注相关性最强的特征，制定针对性的防控措施

六、输出文件
1. 数值型特征相关性分析.csv - 数值型特征的相关性分析
2. 分类型特征影响分析.csv - 分类型特征的影响分析
3. 风险等级特征分析.csv - 风险等级与特征的关系
4. 数值型特征相关性图.png - 相关性可视化
5. 分类型特征概率差异图.png - 概率差异可视化
6. 风险等级与特征关系图.png - 风险等级与特征关系可视化
"""

report_file = f"{output_dir}/风险因素影响分析报告.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"已保存报告: {report_file}")

print("\n" + "="*60)
print("风险因素影响分析完成！")
print("="*60)
print("生成的文件:")
print("1. 数值型特征相关性分析.csv")
print("2. 分类型特征影响分析.csv")
print("3. 风险等级特征分析.csv")
print("4. 数值型特征相关性图.png")
print("5. 分类型特征概率差异图.png")
print("6. 风险等级与特征关系图.png")
print("7. 风险因素影响分析报告.txt")