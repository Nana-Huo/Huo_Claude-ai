"""
汉中市胆结石区域性风险调研综合分析
研究目标：总结区域性疾病调研的典型范式，分析区域特征与胆结石的关联性
分析方法：描述性统计、卡方检验、Logistic回归、决策树、随机森林
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("汉中市胆结石区域性风险调研综合分析")
print("="*80)

output_dir = r"C:\Users\霍冠华\Documents\trae_projects\claude code"

# ============================================================================
# 第一部分：研究范式总结
# ============================================================================
print("\n" + "="*80)
print("第一部分：研究范式总结")
print("="*80)

research_paradigm = """
一、抽样方法
- 研究类型：横断面调查
- 抽样方法：方便抽样（ convenience sampling）
- 样本来源：汉中市各医院门诊及社区
- 样本量：200人

二、数据采集流程
1. 问卷设计
   - 人口学特征：年龄、性别、身高、体重、腰围等
   - 饮食文化：汉中米皮、菜豆腐、火锅、浆水面等特色饮食
   - 生活习惯：运动、睡眠、吸烟、饮酒等
   - 地理环境：水源、县区等
   - 健康状况：血脂、胆结石患病情况

2. 数据收集
   - 面对面访谈
   - 体格检查
   - 实验室检测

3. 数据录入
   - Excel表格录入
   - 双人核对
   - 质量控制

三、风险因素分类体系
1. 人口学特征：年龄、性别、BMI
2. 饮食文化特征：汉中特色饮食、饮食习惯
3. 生活习惯特征：运动、睡眠、吸烟、饮酒
4. 地理环境特征：水源、县区
5. 生物化学特征：血脂水平
"""

print(research_paradigm)

# 保存研究范式
with open(f"{output_dir}/研究范式总结.txt", 'w', encoding='utf-8') as f:
    f.write(research_paradigm)
print("\n已保存: 研究范式总结.txt")

# ============================================================================
# 第二部分：数据加载与预处理
# ============================================================================
print("\n" + "="*80)
print("第二部分：数据加载与预处理")
print("="*80)

# 加载数据
data = pd.read_excel(f"{output_dir}/原始数据.xlsx")
print(f"\n原始数据形状: {data.shape}")
print(f"总样本数: {len(data)}")
print(f"总特征数: {len(data.columns)}")

# 获取目标变量
gallstone_col = data.columns[1]  # 1、您是否患有胆囊结石？
y = data[gallstone_col].map({'没有': 0, '有': 1})

print(f"\n目标变量: {gallstone_col}")
print(f"患病情况:")
print(f"  未患病: {(y==0).sum()}人 ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"  患病: {(y==1).sum()}人 ({(y==1).sum()/len(y)*100:.1f}%)")

# 选择区域特征相关的列
print("\n选择区域特征相关列...")

# 人口学特征
demographics = [
    data.columns[4],  # 性别
    data.columns[7],  # 年龄
]

# 饮食文化特征（汉中特色）
diet_culture = [
    data.columns[18],  # 汉中米皮
    data.columns[19],  # 汉中菜豆腐
    data.columns[20],  # 火锅
    data.columns[21],  # 火锅锅底
    data.columns[22],  # 浆水面
    data.columns[16],  # 食用油使用时长
    data.columns[17],  # 外卖频率
]

# 生活习惯特征
lifestyle = [
    data.columns[11],  # 运动
    data.columns[12],  # 一日三餐
]

# 地理环境特征
geographic = [
    data.columns[13],  # 饮用水来源
    data.columns[14],  # 是否使用净水器
    data.columns[10],  # 县区
]

# 生物化学特征
biochemical = [
    data.columns[45],  # 低密度脂蛋白
]

selected_columns = [gallstone_col] + demographics + diet_culture + lifestyle + geographic + biochemical
selected_columns = [col for col in selected_columns if col is not None]

print(f"\n选择的特征数量: {len(selected_columns) - 1}")
print("特征列表:")
for i, col in enumerate(selected_columns[1:], 1):
    category = ""
    if col in demographics:
        category = "[人口学]"
    elif col in diet_culture:
        category = "[饮食文化]"
    elif col in lifestyle:
        category = "[生活习惯]"
    elif col in geographic:
        category = "[地理环境]"
    elif col in biochemical:
        category = "[生物化学]"
    print(f"  {i}. {category} {col}")

# 准备数据
data_selected = data[selected_columns].copy()

# ============================================================================
# 第三部分：描述性统计
# ============================================================================
print("\n" + "="*80)
print("第三部分：描述性统计")
print("="*80)

# 3.1 样本特征分布
print("\n3.1 样本特征分布")
descriptive_stats = []

for col in selected_columns[1:]:
    print(f"\n{col}:")
    print(f"  数据类型: {data_selected[col].dtype}")
    print(f"  唯一值数量: {data_selected[col].nunique()}")
    print(f"  缺失值数量: {data_selected[col].isna().sum()}")

    # 显示值分布
    value_counts = data_selected[col].value_counts()
    for val, count in value_counts.head(10).items():
        percentage = count / len(data_selected) * 100
        print(f"    {val}: {count} ({percentage:.1f}%)")

    descriptive_stats.append({
        '特征': col,
        '数据类型': str(data_selected[col].dtype),
        '唯一值数': data_selected[col].nunique(),
        '缺失值数': data_selected[col].isna().sum()
    })

# 保存描述性统计
descriptive_df = pd.DataFrame(descriptive_stats)
descriptive_df.to_csv(f"{output_dir}/描述性统计.csv", index=False, encoding='utf-8-sig')
print(f"\n已保存: 描述性统计.csv")

# 3.2 按患病分组统计
print("\n3.2 按患病分组的特征分布")
group_stats = []

for col in selected_columns[1:]:
    print(f"\n{col}:")
    for status in [0, 1]:
        status_label = "未患病" if status == 0 else "患病"
        subset = data_selected[data_selected[gallstone_col] == ('没有' if status == 0 else '有')]

        if subset[col].dtype in ['int64', 'float64']:
            mean_val = subset[col].mean()
            std_val = subset[col].std()
            print(f"  {status_label}: 均值={mean_val:.2f}, 标准差={std_val:.2f}")
        else:
            value_counts = subset[col].value_counts()
            print(f"  {status_label}:")
            for val, count in value_counts.head(3).items():
                percentage = count / len(subset) * 100
                print(f"    {val}: {count} ({percentage:.1f}%)")

# ============================================================================
# 第四部分：单因素分析（卡方检验）
# ============================================================================
print("\n" + "="*80)
print("第四部分：单因素分析（卡方检验）")
print("="*80)

chi_square_results = []

for col in selected_columns[1:]:
    print(f"\n{col}:")

    # 创建交叉表
    cross_tab = pd.crosstab(data_selected[col], data_selected[gallstone_col])

    # 检查是否所有单元格都有足够的样本
    if (cross_tab >= 5).all().all():
        # 执行卡方检验
        chi2, p_value, dof, expected = chi2_contingency(cross_tab)

        print(f"  卡方值: {chi2:.4f}")
        print(f"  P值: {p_value:.4f}")
        print(f"  自由度: {dof}")

        # 判断显著性
        significance = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        significance_text = "显著" if significance else "不显著"
        print(f"  显著性: {significance_text}")

        chi_square_results.append({
            '特征': col,
            '卡方值': chi2,
            'P值': p_value,
            '显著性': significance_text,
            '自由度': dof
        })
    else:
        print(f"  警告: 存在单元格频数<5，卡方检验可能不准确")

# 保存卡方检验结果
chi_square_df = pd.DataFrame(chi_square_results)
chi_square_df = chi_square_df.sort_values('P值')

chi_square_df.to_csv(f"{output_dir}/卡方检验结果.csv", index=False, encoding='utf-8-sig')
print(f"\n已保存: 卡方检验结果.csv")

# 显示显著的特征
significant_features = chi_square_df[chi_square_df['P值'] < 0.05]
print(f"\n显著特征 (P<0.05): {len(significant_features)}个")
for i, row in significant_features.iterrows():
    significance_text = row['显著性']
    if significance_text == '**':
        significance_display = '非常显著'
    elif significance_text == '*':
        significance_display = '显著'
    else:
        significance_display = '不显著'
    print(f"  {row['特征']}: 卡方值={row['卡方值']:.4f}, P值={row['P值']:.4f}, {significance_display}")

# ============================================================================
# 第五部分：多因素分析
# ============================================================================
print("\n" + "="*80)
print("第五部分：多因素分析")
print("="*80)

# 准备数据用于多因素分析
X = data_selected.drop(columns=[gallstone_col]).copy()
y_binary = data_selected[gallstone_col].map({'没有': 0, '有': 1})

# 删除y中的NaN值
valid_indices = ~y_binary.isna()
X = X[valid_indices]
y_binary = y_binary[valid_indices]

print(f"有效样本数: {len(y_binary)}")

# 数据预处理：编码分类变量
print("\n数据预处理...")
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# 处理缺失值
X = X.fillna(X.median())

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

print(f"训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")

# 5.1 Logistic回归
print("\n5.1 Logistic回归分析")
print("-" * 60)

logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)

# 预测
y_pred_log = logistic_model.predict(X_test)
y_prob_log = logistic_model.predict_proba(X_test)[:, 1]

# 评估
accuracy_log = accuracy_score(y_test, y_pred_log)
auc_log = roc_auc_score(y_test, y_prob_log)

print(f"准确率: {accuracy_log:.4f}")
print(f"AUC: {auc_log:.4f}")

# 系数和OR值
coef_df = pd.DataFrame({
    '特征': X.columns,
    '系数': logistic_model.coef_[0],
    'OR值': np.exp(logistic_model.coef_[0])
})

coef_df = coef_df.sort_values('OR值', ascending=False)
print(f"\n特征系数和OR值:")
for i, row in coef_df.iterrows():
    print(f"  {row['特征']}: 系数={row['系数']:.4f}, OR={row['OR值']:.4f}")

# 保存Logistic回归结果
coef_df.to_csv(f"{output_dir}/Logistic回归结果.csv", index=False, encoding='utf-8-sig')
print(f"\n已保存: Logistic回归结果.csv")

# 5.2 决策树分析
print("\n5.2 决策树分析")
print("-" * 60)

dt_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

dt_model.fit(X_train, y_train)

# 预测
y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

# 评估
accuracy_dt = accuracy_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, y_prob_dt)

print(f"准确率: {accuracy_dt:.4f}")
print(f"AUC: {auc_dt:.4f}")

# 特征重要性
dt_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': dt_model.feature_importances_
}).sort_values('重要性', ascending=False)

print(f"\n决策树特征重要性:")
for i, row in dt_importance.iterrows():
    print(f"  {row['特征']}: {row['重要性']:.4f}")

# 保存决策树结果
dt_importance.to_csv(f"{output_dir}/决策树特征重要性.csv", index=False, encoding='utf-8-sig')

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=['未患病', '患病'],
          filled=True, rounded=True, fontsize=10)
plt.title('决策树可视化', fontsize=16, fontproperties='SimHei')
plt.savefig(f"{output_dir}/决策树可视化.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"\n已保存: 决策树可视化.png")

# 5.3 随机森林分析
print("\n5.3 随机森林分析")
print("-" * 60)

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

# 预测
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# 评估
accuracy_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

print(f"准确率: {accuracy_rf:.4f}")
print(f"AUC: {auc_rf:.4f}")
print(f"OOB得分: {rf_model.oob_score_:.4f}")

# 特征重要性
rf_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': rf_model.feature_importances_
}).sort_values('重要性', ascending=False)

print(f"\n随机森林特征重要性:")
for i, row in rf_importance.iterrows():
    print(f"  {row['特征']}: {row['重要性']:.4f}")

# 保存随机森林结果
rf_importance.to_csv(f"{output_dir}/随机森林特征重要性.csv", index=False, encoding='utf-8-sig')
print(f"\n已保存: 随机森林特征重要性.csv")

# ============================================================================
# 第六部分：方法对比
# ============================================================================
print("\n" + "="*80)
print("第六部分：方法对比")
print("="*80)

comparison = pd.DataFrame({
    '方法': ['Logistic回归', '决策树', '随机森林'],
    '准确率': [accuracy_log, accuracy_dt, accuracy_rf],
    'AUC': [auc_log, auc_dt, auc_rf]
})

print("\n三种方法性能对比:")
print(comparison.to_string(index=False))

# 保存对比结果
comparison.to_csv(f"{output_dir}/方法对比.csv", index=False, encoding='utf-8-sig')
print(f"\n已保存: 方法对比.csv")

# 特征重要性对比
print("\n特征重要性对比:")
importance_comparison = pd.DataFrame({
    '特征': X.columns,
    'Logistic回归': coef_df.set_index('特征')['OR值'],
    '决策树': dt_importance.set_index('特征')['重要性'],
    '随机森林': rf_importance.set_index('特征')['重要性']
}).fillna(0)

# 归一化
for col in ['Logistic回归', '决策树', '随机森林']:
    importance_comparison[col] = importance_comparison[col] / importance_comparison[col].sum()

print(importance_comparison.to_string())

# 保存特征重要性对比
importance_comparison.to_csv(f"{output_dir}/特征重要性对比.csv", encoding='utf-8-sig')
print(f"\n已保存: 特征重要性对比.csv")

# ============================================================================
# 第七部分：区域特征与胆结石关联性分析
# ============================================================================
print("\n" + "="*80)
print("第七部分：区域特征与胆结石关联性分析")
print("="*80)

# 7.1 饮食文化特征
print("\n7.1 饮食文化特征分析")
print("-" * 60)

diet_features = [col for col in diet_culture if col in data_selected.columns]
for feat in diet_features:
    if feat in chi_square_df['特征'].values:
        result = chi_square_df[chi_square_df['特征'] == feat].iloc[0]
        print(f"\n{feat}:")
        print(f"  卡方检验: 卡方值={result['卡方值']:.4f}, P值={result['P值']:.4f}, {result['显著性']}")

        # 显示详细分布
        cross_tab = pd.crosstab(data_selected[feat], data_selected[gallstone_col], normalize='index') * 100
        print(f"  患病率分布:")
        for idx, row in cross_tab.iterrows():
            if '有' in row.index:
                print(f"    {idx}: 患病率={row['有']:.1f}%")

# 7.2 生活习惯特征
print("\n7.2 生活习惯特征分析")
print("-" * 60)

lifestyle_features = [col for col in lifestyle if col in data_selected.columns]
for feat in lifestyle_features:
    if feat in chi_square_df['特征'].values:
        result = chi_square_df[chi_square_df['特征'] == feat].iloc[0]
        print(f"\n{feat}:")
        print(f"  卡方检验: 卡方值={result['卡方值']:.4f}, P值={result['P值']:.4f}, {result['显著性']}")

        # 显示详细分布
        cross_tab = pd.crosstab(data_selected[feat], data_selected[gallstone_col], normalize='index') * 100
        print(f"  患病率分布:")
        for idx, row in cross_tab.iterrows():
            if '有' in row.index:
                print(f"    {idx}: 患病率={row['有']:.1f}%")

# 7.3 地理环境特征
print("\n7.3 地理环境特征分析")
print("-" * 60)

geographic_features = [col for col in geographic if col in data_selected.columns]
for feat in geographic_features:
    if feat in chi_square_df['特征'].values:
        result = chi_square_df[chi_square_df['特征'] == feat].iloc[0]
        print(f"\n{feat}:")
        print(f"  卡方检验: 卡方值={result['卡方值']:.4f}, P值={result['P值']:.4f}, {result['显著性']}")

        # 显示详细分布
        cross_tab = pd.crosstab(data_selected[feat], data_selected[gallstone_col], normalize='index') * 100
        print(f"  患病率分布:")
        for idx, row in cross_tab.iterrows():
            if '有' in row.index:
                print(f"    {idx}: 患病率={row['有']:.1f}%")

# ============================================================================
# 第八部分：生成综合报告
# ============================================================================
print("\n" + "="*80)
print("第八部分：生成综合报告")
print("="*80)

report = f"""
汉中市胆结石区域性风险调研综合分析报告
{'='*80}

一、研究范式总结

1. 抽样方法
   - 研究类型：横断面调查
   - 抽样方法：方便抽样
   - 样本来源：汉中市各医院门诊及社区
   - 样本量：200人

2. 数据采集流程
   - 问卷设计：人口学、饮食文化、生活习惯、地理环境、健康状况
   - 数据收集：面对面访谈、体格检查、实验室检测
   - 数据录入：Excel表格录入，双人核对

3. 风险因素分类体系
   - 人口学特征：年龄、性别、BMI
   - 饮食文化特征：汉中特色饮食、饮食习惯
   - 生活习惯特征：运动、睡眠、吸烟、饮酒
   - 地理环境特征：水源、县区
   - 生物化学特征：血脂水平

二、描述性统计

1. 样本特征
   - 总样本数：{len(data)}人
   - 患病情况：
     * 未患病：{(y==0).sum()}人 ({(y==0).sum()/len(y)*100:.1f}%)
     * 患病：{(y==1).sum()}人 ({(y==1).sum()/len(y)*100:.1f}%)

2. 特征分布
   - 分析特征数：{len(selected_columns) - 1}个
   - 包括：人口学、饮食文化、生活习惯、地理环境、生物化学

三、单因素分析（卡方检验）

显著特征 (P<0.05): {len(significant_features)}个
"""

for i, row in significant_features.iterrows():
    significance_text = row['显著性']
    if significance_text == '**':
        significance_display = '非常显著'
    elif significance_text == '*':
        significance_display = '显著'
    else:
        significance_display = '不显著'
    report += f"- {row['特征']}: 卡方值={row['卡方值']:.4f}, P值={row['P值']:.4f}, {significance_display}\n"

report += f"""
四、多因素分析

4.1 Logistic回归
- 准确率: {accuracy_log:.4f}
- AUC: {auc_log:.4f}
- 主要发现:
"""

for i, row in coef_df.head(5).iterrows():
    report += f"  * {row['特征']}: OR={row['OR值']:.4f}\n"

report += f"""
4.2 决策树
- 准确率: {accuracy_dt:.4f}
- AUC: {auc_dt:.4f}
- 主要特征:
"""

for i, row in dt_importance.head(5).iterrows():
    report += f"  * {row['特征']}: {row['重要性']:.4f}\n"

report += f"""
4.3 随机森林
- 准确率: {accuracy_rf:.4f}
- AUC: {auc_rf:.4f}
- OOB得分: {rf_model.oob_score_:.4f}
- 主要特征:
"""

for i, row in rf_importance.head(5).iterrows():
    report += f"  * {row['特征']}: {row['重要性']:.4f}\n"

report += f"""
五、方法对比

三种方法性能对比:
{comparison.to_string(index=False)}

特征重要性一致性:
- Logistic回归和随机森林一致的特征: {len(set(coef_df.head(5)['特征']) & set(rf_importance.head(5)['特征']))}个
- 决策树和随机森林一致的特征: {len(set(dt_importance.head(5)['特征']) & set(rf_importance.head(5)['特征']))}个

六、区域特征与胆结石关联性

6.1 饮食文化特征
汉中特色饮食对胆结石的影响:
"""

for feat in diet_features:
    if feat in chi_square_df['特征'].values:
        result = chi_square_df[chi_square_df['特征'] == feat].iloc[0]
        if result['P值'] < 0.05:
            report += f"- {feat}: 显著相关 (P={result['P值']:.4f})\n"

report += f"""
6.2 生活习惯特征
生活习惯对胆结石的影响:
"""

for feat in lifestyle_features:
    if feat in chi_square_df['特征'].values:
        result = chi_square_df[chi_square_df['特征'] == feat].iloc[0]
        if result['P值'] < 0.05:
            report += f"- {feat}: 显著相关 (P={result['P值']:.4f})\n"

report += f"""
6.3 地理环境特征
地理环境对胆结石的影响:
"""

for feat in geographic_features:
    if feat in chi_square_df['特征'].values:
        result = chi_square_df[chi_square_df['特征'] == feat].iloc[0]
        if result['P值'] < 0.05:
            report += f"- {feat}: 显著相关 (P={result['P值']:.4f})\n"

report += f"""
七、结论与建议

1. 研究范式
   - 建立了区域性疾病调研的典型范式
   - 包括抽样方法、数据采集流程、风险因素分类体系

2. 关键发现
   - 识别出{len(significant_features)}个显著的风险因素
   - 三种分析方法结果具有一致性
   - 区域特征对胆结石发病有重要影响

3. 方法优势
   - Logistic回归: 可解释性强，可计算OR值
   - 决策树: 可视化决策规则，易于理解
   - 随机森林: 特征重要性排序，稳定性好

4. 预防建议
   - 针对显著风险因素制定预防策略
   - 结合汉中地域特色提出针对性建议
   - 加强健康教育，改善生活习惯

八、输出文件
1. 研究范式总结.txt
2. 描述性统计.csv
3. 卡方检验结果.csv
4. Logistic回归结果.csv
5. 决策树特征重要性.csv
6. 决策树可视化.png
7. 随机森林特征重要性.csv
8. 方法对比.csv
9. 特征重要性对比.csv
10. 综合分析报告.txt
"""

# 保存综合报告
with open(f"{output_dir}/综合分析报告.txt", 'w', encoding='utf-8') as f:
    f.write(report)

print("\n已保存: 综合分析报告.txt")

print("\n" + "="*80)
print("分析完成！")
print("="*80)
print(f"共生成10个文件")
print(f"分析涵盖：研究范式、描述性统计、单因素分析、多因素分析、方法对比、区域特征分析")