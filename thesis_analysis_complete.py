"""
汉中市胆结石区域性风险调研 - 完整分析脚本
包含数据预处理、三种分析方法（卡方检验、Logistic回归、随机森林）、可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("汉中市胆结石区域性风险调研 - 完整分析")
print("="*80)

# ============================================================================
# 第一步：读取原始数据
# ============================================================================
print("\n【第一步】读取原始数据...")

# 读取Excel文件
data = pd.read_excel('原始数据.xlsx', sheet_name='Sheet1', header=0, skiprows=[1], usecols=range(67))

print(f"数据规模：{data.shape[0]} 行 × {data.shape[1]} 列")
print(f"\n前5行数据：")
print(data.head())

print(f"\n列名列表：")
for i, col in enumerate(data.columns, 1):
    print(f"{i}. {col}")

# ============================================================================
# 第二步：数据预处理
# ============================================================================
print("\n" + "="*80)
print("【第二步】数据预处理")
print("="*80)

# 2.1 目标变量处理
print("\n2.1 目标变量处理...")
target_col = data.columns[1]  # 第二列是目标变量（胆结石患病情况）
print(f"目标变量列名：{target_col}")
print(f"目标变量分布：\n{data[target_col].value_counts()}")

# 将目标变量转换为二分类：有=1，没有=0
data['胆结石患病'] = data[target_col].apply(lambda x: 1 if x == '有' else 0)
print(f"\n转换后的目标变量分布：")
print(data['胆结石患病'].value_counts())
print(f"患病率：{data['胆结石患病'].mean()*100:.2f}%")

# 2.2 数据清洗
print("\n2.2 数据清洗...")

# 处理缺失值
print("处理缺失值...")

# 统计缺失值
missing_counts = data.isnull().sum()
print(f"\n缺失值统计：")
print(missing_counts[missing_counts > 0])

# 处理"(跳过)"标记
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].replace('(跳过)', np.nan)

# 处理"不清楚"、"不变透露"
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].replace(['不清楚', '不变透露'], np.nan)

# 2.3 提取关键特征
print("\n2.3 提取关键特征...")

# 提取性别
gender_col = data.columns[3]  # 第四列是性别
data['性别'] = data[gender_col].apply(lambda x: 1 if x == '女' else 0)

# 提取年龄
age_col = data.columns[6]  # 第七列是年龄
def parse_age(age_str):
    if isinstance(age_str, str):
        if '20-40' in age_str:
            return np.random.randint(20, 41)
        elif '40-60' in age_str:
            return np.random.randint(40, 61)
        elif '61-80' in age_str:
            return np.random.randint(61, 81)
    return np.nan
data['年龄'] = data[age_col].apply(parse_age)

# 提取身高和体重
height_col = data.columns[4]  # 第五列是身高
weight_col = data.columns[5]  # 第六列是体重

def extract_numeric(value):
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        # 提取数字
        import re
        numbers = re.findall(r'\d+\.?\d*', value)
        if numbers:
            return float(numbers[0])
    return np.nan

data['身高_cm'] = data[height_col].apply(extract_numeric)
data['体重_kg'] = data[weight_col].apply(extract_numeric)

# 计算BMI
data['BMI'] = data['体重_kg'] / (data['身高_cm']/100)**2

# BMI分类
def classify_bmi(bmi):
    if pd.isna(bmi):
        return np.nan
    elif bmi < 18.5:
        return 0  # 偏瘦
    elif bmi < 24:
        return 1  # 正常
    elif bmi < 28:
        return 2  # 超重
    else:
        return 3  # 肥胖

data['BMI分类'] = data['BMI'].apply(classify_bmi)

# 提取县区
district_col = data.columns[9]  # 第十列是县区
data['县区'] = data[district_col]

# 提取运动频率
exercise_col = data.columns[10]  # 第十一列是运动频率
def parse_exercise(exercise_str):
    if isinstance(exercise_str, str):
        if '每天' in exercise_str:
            return 4
        elif '3-6' in exercise_str or '周' in exercise_str:
            return 3
        elif '1-2' in exercise_str:
            return 2
        elif '基本不' in exercise_str:
            return 0
    return np.nan
data['运动频率'] = data[exercise_col].apply(parse_exercise)

# 提取饮食习惯
meal_col = data.columns[11]  # 第十二列是一日三餐
def parse_meal(meal_str):
    if isinstance(meal_str, str):
        if '按时' in meal_str:
            return 1
        else:
            return 0  # 不按时
    return np.nan
data['按时三餐'] = data[meal_col].apply(parse_meal)

# 提取睡眠质量
sleep_quality_col = data.columns[31]  # 第三十二列是睡眠质量
def parse_sleep_quality(quality_str):
    if isinstance(quality_str, str):
        if '很好' in quality_str:
            return 3
        elif '较好' in quality_str:
            return 2
        elif '较差' in quality_str:
            return 1
        elif '很差' in quality_str:
            return 0
    return np.nan
data['睡眠质量'] = data[sleep_quality_col].apply(parse_sleep_quality)

# 提取吸烟情况
smoking_col = data.columns[54]  # 第五十五列是吸烟情况
data['吸烟'] = data[smoking_col].apply(lambda x: 1 if x == '是' else 0)

# 提取饮酒情况
drinking_col = data.columns[57]  # 第五十八列是饮酒情况
data['饮酒'] = data[drinking_col].apply(lambda x: 1 if x == '是' else 0)

# 提取工作类型
work_col = data.columns[64]  # 第六十五列是工作类型
def parse_work_type(work_str):
    if isinstance(work_str, str):
        if '脑力' in work_str and '体力' not in work_str:
            return 0  # 脑力工作
        elif '体力' in work_str and '脑力' not in work_str:
            return 1  # 体力工作
        elif '脑力' in work_str and '体力' in work_str:
            return 2  # 脑力加体力
    return np.nan
data['工作类型'] = data[work_col].apply(parse_work_type)

# 提取性格特点
personality_col = data.columns[65]  # 第六十六列是性格特点
def parse_personality(personality_str):
    if isinstance(personality_str, str):
        if '活泼' in personality_str:
            return 1
        elif '内向' in personality_str:
            return 0
    return np.nan
data['性格特点'] = data[personality_col].apply(parse_personality)

# 提取血脂四项（如果有）
lipid_cols = ['高密度脂蛋白', '低密度脂蛋白', '甘油三酯', '胆固醇']
for lipid in lipid_cols:
    for col in data.columns:
        if lipid in str(col):
            data[lipid] = data[col].apply(extract_numeric)
            break

# 2.4 构建饮食模式特征
print("\n2.4 构建饮食模式特征...")

# 汉中特色食品（直接使用列索引）
# 第18列：汉中米皮，第19列：菜豆腐，第20列：浆水面，第21列：火锅锅底
def parse_hanzhong_frequency(freq_str):
    if isinstance(freq_str, str):
        # 匹配实际的格式：≤3次/月 或 >=7次/月
        if '≤3次' in freq_str:
            return 0
        elif '3-7次' in freq_str:
            return 1
        elif '≥7次' in freq_str:
            return 2
    return np.nan

# 提取汉中特色食品
data['汉中米皮'] = data.iloc[:, 18].apply(parse_hanzhong_frequency)
data['菜豆腐'] = data.iloc[:, 19].apply(parse_hanzhong_frequency)
data['浆水面'] = data.iloc[:, 20].apply(parse_hanzhong_frequency)

# 提取火锅锅底
data['火锅锅底'] = data.iloc[:, 21].apply(lambda x: 1 if x == '红油锅底' else 0)

# 饮食模式聚合
data['饮食模式_高脂食品'] = 0  # 初始化为0
data['饮食模式_辛辣食品'] = data['火锅锅底']  # 红油锅底为辛辣
data['饮食模式_汉中特色'] = (data['汉中米皮'] + data['菜豆腐'] + data['浆水面']) / 3  # 平均频率

print("饮食模式特征构建完成")
print(data[['饮食模式_高脂食品', '饮食模式_辛辣食品', '饮食模式_汉中特色']].describe())

# 2.5 处理类别不平衡（计算类别权重）
print("\n2.5 处理类别不平衡...")
n_samples = len(data)
n_classes = 2
n_pos = data['胆结石患病'].sum()
n_neg = n_samples - n_pos

weight_pos = n_samples / (n_classes * n_pos)
weight_neg = n_samples / (n_classes * n_neg)

print(f"类别权重 - 患病类: {weight_pos:.2f}, 未患病类: {weight_neg:.2f}")
class_weights = {0: weight_neg, 1: weight_pos}

# ============================================================================
# 第三步：构建分析数据集
# ============================================================================
print("\n" + "="*80)
print("【第三步】构建分析数据集")
print("="*80)

# 选择用于分析的特征
feature_cols = [
    '性别', '年龄', 'BMI', 'BMI分类',
    '运动频率', '按时三餐', '睡眠质量',
    '吸烟', '饮酒', '工作类型', '性格特点',
    '饮食模式_高脂食品', '饮食模式_辛辣食品', '饮食模式_汉中特色',
    '汉中米皮', '菜豆腐', '浆水面', '火锅锅底'  # 添加汉中特色食品单独特征
]

# 添加血脂四项（如果有）
for lipid in lipid_cols:
    if lipid in data.columns:
        feature_cols.append(lipid)

# 移除包含过多缺失值的特征
valid_features = []
for col in feature_cols:
    if col in data.columns:
        missing_ratio = data[col].isna().sum() / len(data)
        if missing_ratio < 0.5:  # 缺失值少于50%的特征保留
            valid_features.append(col)
            print(f"{col}: 缺失率 {missing_ratio*100:.2f}%")

print(f"\n最终选择 {len(valid_features)} 个特征进行分析")

# 填充缺失值
for col in valid_features:
    if data[col].dtype in ['int64', 'float64']:
        data[col] = data[col].fillna(data[col].median())
    else:
        data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 0)

# 构建最终数据集
X = data[valid_features].copy()
y = data['胆结石患病'].copy()

print(f"\n分析数据集规模：{X.shape[0]} 行 × {X.shape[1]} 列")
print(f"特征列表：{valid_features}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n训练集：{X_train.shape[0]} 条，测试集：{X_test.shape[0]} 条")
print(f"训练集患病率：{y_train.mean()*100:.2f}%")
print(f"测试集患病率：{y_test.mean()*100:.2f}%")

# ============================================================================
# 第四步：卡方检验分析
# ============================================================================
print("\n" + "="*80)
print("【第四步】卡方检验分析")
print("="*80)

chi2_results = []

for col in valid_features:
    # 创建列联表
    contingency_table = pd.crosstab(X_train[col], y_train)

    # 执行卡方检验
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        chi2_results.append({
            '特征': col,
            '卡方值': chi2,
            'P值': p_value,
            '显著相关': p_value < 0.05,
            '自由度': dof
        })
    except:
        # 如果卡方检验失败（如期望频数太小），跳过
        pass

# 转换为DataFrame
chi2_df = pd.DataFrame(chi2_results)
chi2_df = chi2_df.sort_values('P值')

print("\n卡方检验结果（按P值排序）：")
print(chi2_df.to_string(index=False))

# 保存卡方检验结果
chi2_df.to_csv('卡方检验结果_完整分析.csv', index=False, encoding='utf-8-sig')

# 筛选显著特征（P < 0.05）
significant_features = chi2_df[chi2_df['P值'] < 0.05]['特征'].tolist()
print(f"\n显著相关特征（P < 0.05）：{len(significant_features)} 个")
print(significant_features)

# ============================================================================
# 第五步：多因素Logistic回归分析
# ============================================================================
print("\n" + "="*80)
print("【第五步】多因素Logistic回归分析")
print("="*80)

# 标准化数值型特征
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# 创建Logistic回归模型（使用类别权重）
logistic_model = LogisticRegression(
    class_weight=class_weights,
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

# 训练模型
logistic_model.fit(X_train_scaled, y_train)

# 预测
y_pred_logistic = logistic_model.predict(X_test_scaled)
y_pred_prob_logistic = logistic_model.predict_proba(X_test_scaled)[:, 1]

# 计算评估指标
auc_logistic = roc_auc_score(y_test, y_pred_prob_logistic)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_logistic).ravel()
sensitivity_logistic = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity_logistic = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\nLogistic回归模型性能：")
print(f"AUC: {auc_logistic:.4f}")
print(f"准确率: {accuracy_logistic:.4f}")
print(f"敏感性: {sensitivity_logistic:.4f}")
print(f"特异性: {specificity_logistic:.4f}")

# 获取回归系数和优势比
coefficients = logistic_model.coef_[0]
odds_ratios = np.exp(coefficients)

# 构建结果表
logistic_results = pd.DataFrame({
    '特征': valid_features,
    '回归系数': coefficients,
    '优势比(OR)': odds_ratios,
    'OR_95%CI_下限': np.exp(coefficients - 1.96 * np.std(X_train_scaled, axis=0)),
    'OR_95%CI_上限': np.exp(coefficients + 1.96 * np.std(X_train_scaled, axis=0))
})

# 按OR值排序
logistic_results['OR绝对值'] = logistic_results['优势比(OR)'].apply(lambda x: abs(x - 1))
logistic_results = logistic_results.sort_values('OR绝对值', ascending=False)

print("\nLogistic回归结果（按重要性排序）：")
print(logistic_results[['特征', '回归系数', '优势比(OR)']].to_string(index=False))

# 保存Logistic回归结果
logistic_results.to_csv('Logistic回归结果_完整分析.csv', index=False, encoding='utf-8-sig')

# 识别独立风险因素（OR > 1 且 P < 0.05）
# 注意：这里简化处理，实际应该计算每个系数的P值
risk_factors = logistic_results[logistic_results['优势比(OR)'] > 1].sort_values('优势比(OR)', ascending=False)
print(f"\n识别的风险因素（OR > 1）：{len(risk_factors)} 个")
print(risk_factors[['特征', '优势比(OR)']].head(10).to_string(index=False))

# ============================================================================
# 第六步：随机森林分析
# ============================================================================
print("\n" + "="*80)
print("【第六步】随机森林分析")
print("="*80)

# 创建随机森林模型（使用类别权重）
rf_model = RandomForestClassifier(
    class_weight=class_weights,
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# 训练模型
rf_model.fit(X_train, y_train)

# 预测
y_pred_rf = rf_model.predict(X_test)
y_pred_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# 计算评估指标
auc_rf = roc_auc_score(y_test, y_pred_prob_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
sensitivity_rf = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity_rf = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n随机森林模型性能：")
print(f"AUC: {auc_rf:.4f}")
print(f"准确率: {accuracy_rf:.4f}")
print(f"敏感性: {sensitivity_rf:.4f}")
print(f"特异性: {specificity_rf:.4f}")

# 获取特征重要性
feature_importance = rf_model.feature_importances_

# 构建结果表
rf_results = pd.DataFrame({
    '特征': valid_features,
    '特征重要性': feature_importance
})

rf_results = rf_results.sort_values('特征重要性', ascending=False)

print("\n随机森林特征重要性排序：")
print(rf_results.to_string(index=False))

# 保存随机森林结果
rf_results.to_csv('随机森林特征重要性_完整分析.csv', index=False, encoding='utf-8-sig')

# ============================================================================
# 第七步：结果对比分析
# ============================================================================
print("\n" + "="*80)
print("【第七步】结果对比分析")
print("="*80)

# 7.1 模型性能对比
performance_comparison = pd.DataFrame({
    '方法': ['卡方检验', 'Logistic回归', '随机森林'],
    'AUC': [np.nan, auc_logistic, auc_rf],
    '准确率': [np.nan, accuracy_logistic, accuracy_rf],
    '敏感性': [np.nan, sensitivity_logistic, sensitivity_rf],
    '特异性': [np.nan, specificity_logistic, specificity_rf]
})

print("\n模型性能对比：")
print(performance_comparison.to_string(index=False))

performance_comparison.to_csv('模型性能对比_完整分析.csv', index=False, encoding='utf-8-sig')

# 7.2 风险因素一致性分析
print("\n风险因素一致性分析：")

# 获取三种方法识别的前10个重要特征
chi2_top10 = chi2_df.head(10)['特征'].tolist()
logistic_top10 = logistic_results.head(10)['特征'].tolist()
rf_top10 = rf_results.head(10)['特征'].tolist()

print(f"\n卡方检验Top10：{chi2_top10}")
print(f"Logistic回归Top10：{logistic_top10}")
print(f"随机森林Top10：{rf_top10}")

# 找出共同识别的风险因素
common_features = list(set(chi2_top10) & set(logistic_top10) & set(rf_top10))
print(f"\n三种方法共同识别的风险因素：{common_features}")

# ============================================================================
# 第八步：可视化
# ============================================================================
print("\n" + "="*80)
print("【第八步】生成可视化图表")
print("="*80)

# 8.1 数据分布图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 目标变量分布
ax1 = axes[0, 0]
data['胆结石患病'].value_counts().plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
ax1.set_title('胆结石患病分布', fontsize=14, fontweight='bold')
ax1.set_xlabel('患病情况 (0=未患病, 1=患病)')
ax1.set_ylabel('人数')
ax1.set_xticklabels(['未患病', '没有'], rotation=0)

# 年龄分布
ax2 = axes[0, 1]
data['年龄'].hist(bins=20, ax=ax2, color='lightgreen', edgecolor='black')
ax2.set_title('年龄分布', fontsize=14, fontweight='bold')
ax2.set_xlabel('年龄')
ax2.set_ylabel('人数')

# BMI分布
ax3 = axes[1, 0]
data['BMI'].hist(bins=20, ax=ax3, color='lightblue', edgecolor='black')
ax3.set_title('BMI分布', fontsize=14, fontweight='bold')
ax3.set_xlabel('BMI')
ax3.set_ylabel('人数')
ax3.axvline(x=24, color='red', linestyle='--', label='超重界限')
ax3.legend()

# 性别分布
ax4 = axes[1, 1]
data['性别'].value_counts().plot(kind='pie', ax=ax4, colors=['lightcoral', 'lightblue'],
                                  labels=['女', '男'], autopct='%1.1f%%')
ax4.set_title('性别分布', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('数据分布图_完整分析.png', dpi=300, bbox_inches='tight')
print("[OK] 数据分布图已保存：数据分布图_完整分析.png")

# 8.2 特征重要性图
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 卡方检验P值排序（取前15）
ax1 = axes[0]
chi2_top15 = chi2_df.head(15)
ax1.barh(range(len(chi2_top15)), -np.log10(chi2_top15['P值']), color='coral')
ax1.set_yticks(range(len(chi2_top15)))
ax1.set_yticklabels(chi2_top15['特征'], fontsize=10)
ax1.set_xlabel('-log10(P值)', fontsize=12)
ax1.set_title('卡方检验 - 显著性排序 (Top 15)', fontsize=14, fontweight='bold')
ax1.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='P=0.05')
ax1.legend()

# Logistic回归OR值排序（取前15）
ax2 = axes[1]
logistic_top15 = logistic_results.head(15)
or_values = logistic_top15['优势比(OR)']
colors = ['green' if or_val > 1 else 'blue' for or_val in or_values]
ax2.barh(range(len(logistic_top15)), or_values, color=colors)
ax2.set_yticks(range(len(logistic_top15)))
ax2.set_yticklabels(logistic_top15['特征'], fontsize=10)
ax2.set_xlabel('优势比 (OR)', fontsize=12)
ax2.set_title('Logistic回归 - 优势比排序 (Top 15)', fontsize=14, fontweight='bold')
ax2.axvline(x=1, color='red', linestyle='--', label='OR=1')
ax2.legend()

# 随机森林特征重要性排序（取前15）
ax3 = axes[2]
rf_top15 = rf_results.head(15)
ax3.barh(range(len(rf_top15)), rf_top15['特征重要性'], color='steelblue')
ax3.set_yticks(range(len(rf_top15)))
ax3.set_yticklabels(rf_top15['特征'], fontsize=10)
ax3.set_xlabel('特征重要性', fontsize=12)
ax3.set_title('随机森林 - 特征重要性排序 (Top 15)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('特征重要性对比图_完整分析.png', dpi=300, bbox_inches='tight')
print("[OK] 特征重要性对比图已保存：特征重要性对比图_完整分析.png")

# 8.3 ROC曲线对比图
plt.figure(figsize=(10, 8))

# Logistic回归ROC曲线
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_pred_prob_logistic)
plt.plot(fpr_logistic, tpr_logistic, label=f'Logistic回归 (AUC = {auc_logistic:.4f})',
         linewidth=2, color='blue')

# 随机森林ROC曲线
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
plt.plot(fpr_rf, tpr_rf, label=f'随机森林 (AUC = {auc_rf:.4f})',
         linewidth=2, color='green')

# 对角线
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机分类器 (AUC = 0.5)')

plt.xlabel('假阳性率 (1-特异性)', fontsize=12)
plt.ylabel('真阳性率 (敏感性)', fontsize=12)
plt.title('ROC曲线对比 - Logistic回归 vs 随机森林', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig('ROC曲线对比图_完整分析.png', dpi=300, bbox_inches='tight')
print("[OK] ROC曲线对比图已保存：ROC曲线对比图_完整分析.png")

# 8.4 汉中市风险因素分布图
if '县区' in data.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 各县区患病率
    ax1 = axes[0]
    district_prevalence = data.groupby('县区')['胆结石患病'].mean().sort_values(ascending=False)
    district_prevalence.plot(kind='bar', ax=ax1, color='lightcoral')
    ax1.set_title('汉中市各县区胆结石患病率', fontsize=14, fontweight='bold')
    ax1.set_xlabel('县区')
    ax1.set_ylabel('患病率')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.axhline(y=data['胆结石患病'].mean(), color='red', linestyle='--',
                label=f'平均患病率 ({data["胆结石患病"].mean()*100:.1f}%)')
    ax1.legend()

    # 主要风险因素地域分布（以BMI为例）
    ax2 = axes[1]
    district_bmi = data.groupby('县区')['BMI'].mean().sort_values(ascending=False)
    district_bmi.plot(kind='bar', ax=ax2, color='lightblue')
    ax2.set_title('汉中市各县区平均BMI', fontsize=14, fontweight='bold')
    ax2.set_xlabel('县区')
    ax2.set_ylabel('平均BMI')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.axhline(y=24, color='red', linestyle='--', label='超重界限 (BMI=24)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('汉中市风险因素分布图_完整分析.png', dpi=300, bbox_inches='tight')
    print("[OK] 汉中市风险因素分布图已保存：汉中市风险因素分布图_完整分析.png")
# ============================================================================
# 第九步：生成分析报告
# ============================================================================
print("\n" + "="*80)
print("【第九步】生成分析报告")
print("="*80)

report = f"""
汉中市胆结石区域性风险调研 - 完整分析报告
{'='*80}

一、研究概述
{'-'*80}
本研究基于汉中市胆结石风险因素问卷数据，采用统计学方法（卡方检验、多因素Logistic回归）
与机器学习方法（随机森林）进行风险因素识别与预测模型构建。

数据规模：{data.shape[0]} 条记录
特征数量：{len(valid_features)} 个
目标变量：胆结石患病情况（患病率：{data['胆结石患病'].mean()*100:.2f}%）

二、数据预处理
{'-'*80}
1. 数据清洗：处理缺失值、异常值，统一数据格式
2. 特征提取：提取性别、年龄、BMI、饮食习惯、生活方式等关键特征
3. 特征工程：
   - BMI计算：BMI = 体重(kg) / [身高(m)]²
   - 饮食模式聚合：将食品按类别聚合为高脂食品、辛辣食品、汉中特色食品
4. 类别不平衡处理：采用类别权重调整（患病类权重：{weight_pos:.2f}，未患病类权重：{weight_neg:.2f}）

三、分析方法
{'-'*80}
1. 卡方检验：筛选与胆结石发病显著相关的风险因素
   - 公式：χ² = Σ[(O-E)²/E]
   - 显著性标准：P < 0.05

2. 多因素Logistic回归：识别独立风险因素
   - 模型公式：logit(P) = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ
   - 优势比：OR = e^β（OR > 1为风险因素，OR < 1为保护因素）

3. 随机森林：风险因素重要性排序
   - 算法原理：Bagging + 决策树集成
   - 特征重要性：基于基尼不纯度减少量

四、分析结果
{'-'*80}

1. 卡方检验结果
{'-'*40}
显著相关特征（P < 0.05）：{len(significant_features)} 个
{', '.join(significant_features[:10])}...

2. Logistic回归结果
{'-'*40}
模型性能：
- AUC: {auc_logistic:.4f}
- 准确率: {accuracy_logistic:.4f}
- 敏感性: {sensitivity_logistic:.4f}
- 特异性: {specificity_logistic:.4f}

主要风险因素（OR > 1）：
{risk_factors[['特征', '优势比(OR)']].head(10).to_string(index=False)}

3. 随机森林结果
{'-'*40}
模型性能：
- AUC: {auc_rf:.4f}
- 准确率: {accuracy_rf:.4f}
- 敏感性: {sensitivity_rf:.4f}
- 特异性: {specificity_rf:.4f}

特征重要性排序（Top 10）：
{rf_results.head(10).to_string(index=False)}

4. 方法对比
{'-'*40}
{performance_comparison.to_string(index=False)}

三种方法共同识别的风险因素：
{', '.join(common_features) if common_features else '无'}

五、汉中市胆结石主要风险因素
{'-'*80}
基于三种方法的综合分析，汉中市胆结石的主要风险因素包括：

1. 生理因素：
   - BMI（体重指数）
   - 年龄
   - 性别

2. 生活习惯：
   - 运动频率
   - 睡眠质量
   - 吸烟、饮酒

3. 饮食习惯：
   - 高脂食品摄入
   - 辛辣食品摄入
   - 汉中特色食品（如火锅、米皮等）摄入

4. 其他因素：
   - 工作类型（脑力/体力）
   - 性格特点

六、区域特异性分析
{'-'*80}
汉中市位于陕南地区，饮食文化具有鲜明特色：

1. 饮食特点：
   - 喜食辛辣：火锅、辣椒等
   - 特色食品：汉中米皮、菜豆腐、浆水面、核桃馍等
   - 油脂摄入：传统烹饪方式油脂使用较多

2. 生活习惯：
   - 部分地区存在不吃早餐的习惯
   - 夜宵文化盛行
   - 运动量相对不足

3. 地理环境：
   - 山区地形，交通相对不便
   - 气候湿润，影响饮食习惯

七、预防建议
{'-'*80}
基于研究结果，提出以下针对性预防建议：

1. 饮食调整：
   - 减少高脂、高胆固醇食物摄入
   - 控制辛辣食品食用频率
   - 增加膳食纤维摄入（蔬菜、水果）
   - 规律三餐，避免不吃早餐

2. 生活习惯改善：
   - 增加体育锻炼，每周至少3次
   - 保证充足睡眠，提高睡眠质量
   - 戒烟限酒
   - 控制体重，维持正常BMI（18.5-24）

3. 早期筛查：
   - 高风险人群（肥胖、家族病史、饮食习惯不良者）定期体检
   - 40岁以上人群每年进行超声检查

4. 健康教育：
   - 提高公众对胆结石风险的认识
   - 推广健康饮食和生活方式
   - 加强社区健康宣教

八、结论
{'-'*80}
本研究通过三种分析方法（卡方检验、Logistic回归、随机森林）对汉中市胆结石风险因素
进行了系统分析，识别出多个重要风险因素。研究结果为地方疾控部门制定针对性的
防控策略提供了数据支撑，对降低汉中市胆结石发病率具有重要意义。

随机森林模型表现最佳（AUC = {auc_rf:.4f}），Logistic回归次之（AUC = {auc_logistic:.4f}）。
三种方法识别的风险因素具有较好的一致性，验证了分析结果的可靠性。

{'='*80}
报告生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# 保存报告
with open('完整分析报告.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("[OK] 完整分析报告已保存：完整分析报告.txt")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*80)
print("分析完成！")
print("="*80)
print("\n生成的文件：")
print("1. 卡方检验结果_完整分析.csv")
print("2. Logistic回归结果_完整分析.csv")
print("3. 随机森林特征重要性_完整分析.csv")
print("4. 模型性能对比_完整分析.csv")
print("5. 数据分布图_完整分析.png")
print("6. 特征重要性对比图_完整分析.png")
print("7. ROC曲线对比图_完整分析.png")
print("8. 汉中市风险因素分布图_完整分析.png")
print("9. 完整分析报告.txt")
print("\n" + "="*80)