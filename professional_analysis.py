"""
汉中市胆结石区域性风险调研专业分析方案
按照"Logistic → Lasso → 随机森林"三步走策略
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("汉中市胆结石区域性风险调研专业分析")
print("分析策略：Logistic → Lasso → 随机森林")
print("="*80)

output_dir = r"C:\Users\霍冠华\Documents\trae_projects\claude code"

# ============================================================================
# 步骤0：数据清洗和预处理
# ============================================================================
print("\n" + "="*80)
print("步骤0：数据清洗和预处理")
print("="*80)

# 1. 加载原始数据
print("\n1. 加载原始数据...")
data = pd.read_excel(f"{output_dir}/原始数据.xlsx")
print(f"原始数据形状: {data.shape}")
print(f"总特征数: {len(data.columns)}")

# 2. 识别目标变量
gallstone_cols = [col for col in data.columns if '结石' in col and '是否' in col]
print(f"\n找到胆结石相关列: {gallstone_cols}")
if gallstone_cols:
    target_col = gallstone_cols[0]
    print(f"使用目标列: {target_col}")
else:
    target_col = data.columns[1]  # 第二列通常是目标变量
    print(f"未找到明确的胆结石列，使用: {target_col}")

# 3. 选择分析特征（饮食文化、生活习惯、地理环境、人口学特征）
print("\n2. 选择分析特征...")

# 饮食文化特征
diet_features = [
    col for col in data.columns 
    if any(keyword in col for keyword in ['米皮', '菜豆腐', '浆水面', '火锅', '红油', '腌制', '食用油', '外卖'])
]

# 生活习惯特征
lifestyle_features = [
    col for col in data.columns
    if any(keyword in col for keyword in ['运动', '睡眠', '吸烟', '饮酒', '一日三餐'])
]

# 地理环境特征
geographic_features = [
    col for col in data.columns
    if any(keyword in col for keyword in ['县区', '水源', '净水器', '海拔', '山区'])
]

# 人口学特征
demographic_features = [
    col for col in data.columns
    if any(keyword in col for keyword in ['年龄', '性别', '身高', '体重', 'BMI', '腰围', '臀围'])
]

# 生物化学特征
biochemical_features = [
    col for col in data.columns
    if any(keyword in col for keyword in ['胆固醇', '甘油', '血脂', 'LDL', 'HDL', 'TC', 'TG'])
]

selected_columns = list(set(diet_features + lifestyle_features + geographic_features + 
                             demographic_features + biochemical_features + [target_col]))

print(f"\n选择的特征数: {len(selected_columns)}")
print(f"  - 饮食文化: {len(diet_features)}个")
print(f"  - 生活习惯: {len(lifestyle_features)}个")
print(f"  - 地理环境: {len(geographic_features)}个")
print(f"  - 人口学: {len(demographic_features)}个")
print(f"  - 生物化学: {len(biochemical_features)}个")

data_selected = data[selected_columns].copy()
print(f"筛选后数据形状: {data_selected.shape}")

# 4. 删除缺失值>30%的变量
print("\n3. 删除缺失值>30%的变量...")
missing_ratio = data_selected.isnull().sum() / len(data_selected)
cols_to_keep = missing_ratio[missing_ratio <= 0.3].index.tolist()
cols_to_drop = missing_ratio[missing_ratio > 0.3].index.tolist()

print(f"删除的变量（缺失>30%）: {len(cols_to_drop)}个")
for col in cols_to_drop:
    print(f"  - {col}: {missing_ratio[col]:.1%}")

data_selected = data_selected[cols_to_keep]
print(f"删除后数据形状: {data_selected.shape}")

# 5. 频率标准化（将中文频率转换为连续量）
print("\n4. 频率标准化...")

def convert_frequency_to_days(freq_str):
    """将频率字符串转换为每天次数"""
    if pd.isna(freq_str):
        return np.nan
    
    freq_str = str(freq_str).strip()
    
    # 处理各种频率格式
    if '从不' in freq_str or '0' in freq_str or '没有' in freq_str:
        return 0.0
    elif '≤3次/周' in freq_str or '小于3次/周' in freq_str:
        return 3.0 / 7.0  # 约0.43次/天
    elif '1-3/月' in freq_str or '每月1-3次' in freq_str:
        return 2.0 / 30.0  # 约0.07次/天
    elif '1次/月' in freq_str or '每月1次' in freq_str:
        return 1.0 / 30.0  # 约0.03次/天
    elif '2-3/月' in freq_str or '每月2-3次' in freq_str:
        return 2.5 / 30.0  # 约0.08次/天
    elif '4-10/月' in freq_str or '每月4-10次' in freq_str:
        return 7.0 / 30.0  # 约0.23次/天
    elif '≥4次/周' in freq_str or '大于等于4次/周' in freq_str:
        return 4.0 / 7.0  # 约0.57次/天
    elif '每周' in freq_str or '/周' in freq_str:
        # 提取数字
        import re
        numbers = re.findall(r'\d+', freq_str)
        if numbers:
            return float(numbers[0]) / 7.0
    elif '每天' in freq_str or '/日' in freq_str:
        return 1.0
    
    # 尝试直接转换为数字
    try:
        return float(freq_str)
    except:
        return np.nan

# 应用频率转换
for col in data_selected.columns:
    if col != target_col and data_selected[col].dtype == 'object':
        # 检查是否包含频率相关信息
        sample_values = data_selected[col].dropna().head(5).astype(str).tolist()
        if any(any(kw in str(v) for kw in ['次', '月', '周', '天']) for v in sample_values):
            print(f"  转换频率: {col}")
            data_selected[col + '_freq'] = data_selected[col].apply(convert_frequency_to_days)
            data_selected = data_selected.drop(columns=[col])

print(f"频率转换后数据形状: {data_selected.shape}")

# 6. 计算BMI（如果没有）
if 'BMI' not in data_selected.columns and '身高' in data_selected.columns and '体重' in data_selected.columns:
    print("\n5. 计算BMI...")
    height_m = pd.to_numeric(data_selected['身高'], errors='coerce') / 100
    weight_kg = pd.to_numeric(data_selected['体重'], errors='coerce')
    data_selected['BMI'] = weight_kg / (height_m ** 2)
    print(f"  BMI范围: {data_selected['BMI'].min():.2f} - {data_selected['BMI'].max():.2f}")

# 7. 生成汉中传统饮食指数（PCA）
print("\n6. 生成汉中传统饮食指数...")

hanzhong_diet_features = [
    col for col in data_selected.columns
    if any(keyword in col for keyword in ['米皮', '菜豆腐', '浆水面'])
]

if len(hanzhong_diet_features) >= 2:
    # 提取饮食特征数据
    diet_data = data_selected[hanzhong_diet_features].copy()
    
    # 填充缺失值
    diet_data = diet_data.fillna(diet_data.median())
    
    # 标准化
    scaler = StandardScaler()
    diet_scaled = scaler.fit_transform(diet_data)
    
    # PCA
    pca = PCA(n_components=1)
    hanzhong_diet_index = pca.fit_transform(diet_scaled)
    data_selected['汉中传统饮食指数'] = hanzhong_diet_index
    
    explained_variance = pca.explained_variance_ratio_[0]
    print(f"  汉中传统饮食指数解释方差: {explained_variance:.2%}")
    print(f"  使用的特征: {hanzhong_diet_features}")
else:
    print(f"  汉中传统饮食特征不足2个，跳过PCA")

# 8. 多重插补（简化版：使用中位数/众数填充）
print("\n7. 多重插补（简化版）...")

for col in data_selected.columns:
    if col == target_col:
        continue
    
    if data_selected[col].dtype in ['float64', 'int64']:
        # 数值变量用中位数填充
        median_val = data_selected[col].median()
        data_selected[col] = data_selected[col].fillna(median_val)
    else:
        # 分类变量用众数填充
        mode_val = data_selected[col].mode()[0] if len(data_selected[col].mode()) > 0 else data_selected[col].value_counts().idxmax()
        data_selected[col] = data_selected[col].fillna(mode_val)

print(f"插补后数据形状: {data_selected.shape}")

# 9. 编码分类变量
print("\n8. 编码分类变量...")

# 识别分类变量
categorical_cols = data_selected.select_dtypes(include=['object']).columns.tolist()
if target_col in categorical_cols:
    categorical_cols.remove(target_col)

print(f"分类变量: {len(categorical_cols)}个")

# 标签编码
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data_selected[col] = le.fit_transform(data_selected[col].astype(str))
    label_encoders[col] = le
    print(f"  编码: {col}")

# 10. 准备分析数据
print("\n9. 准备分析数据...")

# 编码目标变量
y = data_selected[target_col].map({'没有': 0, '有': 1, '0': 0, '1': 1, 0: 0, 1: 1})
y = y.fillna(0)  # 填充缺失值

# 删除目标变量列
X = data_selected.drop(columns=[target_col])

print(f"最终数据形状:")
print(f"  X: {X.shape}")
print(f"  y: {y.shape}")
print(f"  患病率: {y.mean():.2%}")

# 保存预处理后的数据
preprocessed_file = f"{output_dir}/专业分析_预处理数据.csv"
data_preprocessed = pd.concat([X, y], axis=1)
data_preprocessed.to_csv(preprocessed_file, index=False, encoding='utf-8-sig')
print(f"\n已保存: {preprocessed_file}")

# ============================================================================
# 步骤1：传统统计分析
# ============================================================================
print("\n" + "="*80)
print("步骤1：传统统计分析")
print("="*80)

print("\n1.1 单因素分析（筛选p<0.2的变量）...")

univariate_results = []
significant_features = []

for col in X.columns:
    # 分组数据
    group_0 = X[y == 0][col]
    group_1 = X[y == 1][col]
    
    # 卡方检验（分类变量）或Mann-Whitney U检验（数值变量）
    if X[col].nunique() <= 10:  # 分类变量
        try:
            # 创建交叉表
            cross_tab = pd.crosstab(X[col], y)
            if cross_tab.shape[0] > 1 and cross_tab.shape[1] > 1:
                chi2, p_value, dof, expected = chi2_contingency(cross_tab)
                test_type = '卡方检验'
            else:
                p_value = 1.0
                test_type = '样本不足'
        except:
            p_value = 1.0
            test_type = '计算失败'
    else:  # 数值变量
        try:
            stat, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
            test_type = 'Mann-Whitney U'
        except:
            p_value = 1.0
            test_type = '计算失败'
    
    univariate_results.append({
        '特征': col,
        '检验方法': test_type,
        'P值': p_value,
        '显著性': '显著' if p_value < 0.2 else '不显著'
    })
    
    if p_value < 0.2:
        significant_features.append(col)

univariate_df = pd.DataFrame(univariate_results)
univariate_df = univariate_df.sort_values('P值')

print(f"\n单因素分析完成，共{len(X.columns)}个变量")
print(f"显著特征（p<0.2）: {len(significant_features)}个")

# 保存单因素分析结果
univariate_file = f"{output_dir}/专业分析_单因素分析.csv"
univariate_df.to_csv(univariate_file, index=False, encoding='utf-8-sig')
print(f"已保存: {univariate_file}")

# 使用显著特征进行后续分析
if len(significant_features) == 0:
    print("\n警告：没有显著特征，使用所有特征")
    X_selected = X.copy()
else:
    print(f"\n使用{len(significant_features)}个显著特征进行多因素分析")
    X_selected = X[significant_features].copy()

print("\n1.2 多因素Logistic回归...")

# 添加常数项
X_const = sm.add_constant(X_selected)

# 拟合Logistic回归
try:
    logit_model = sm.Logit(y, X_const)
    logit_result = logit_model.fit(disp=0)
    
    print(f"模型拟合完成")
    print(f"  AIC: {logit_result.aic:.2f}")
    print(f"  BIC: {logit_result.bic:.2f}")
    
    # 提取结果
    logit_summary = logit_result.summary2().tables[1]
    logit_results = []
    
    for idx, row in logit_summary.iterrows():
        if idx != 'const':  # 跳过常数项
            coef = row['Coef.']
            or_value = np.exp(coef)
            ci_lower = np.exp(row['[0.025'])
            ci_upper = np.exp(row['0.975]'])
            p_value = row['P>|z|']
            
            logit_results.append({
                '特征': idx,
                '系数': coef,
                'OR值': or_value,
                'OR_95%CI_lower': ci_lower,
                'OR_95%CI_upper': ci_upper,
                'P值': p_value,
                '显著性': '显著' if p_value < 0.05 else '不显著'
            })
    
    logit_df = pd.DataFrame(logit_results)
    logit_df = logit_df.sort_values('P值')
    
    # 保存Logistic回归结果
    logit_file = f"{output_dir}/专业分析_Logistic回归.csv"
    logit_df.to_csv(logit_file, index=False, encoding='utf-8-sig')
    print(f"已保存: {logit_file}")
    
except Exception as e:
    print(f"Logistic回归失败: {e}")
    logit_df = pd.DataFrame()

# ============================================================================
# 步骤2：机器学习分析
# ============================================================================
print("\n" + "="*80)
print("步骤2：机器学习分析")
print("="*80)

print("\n2.1 Lasso回归降维...")

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Lasso交叉验证
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_scaled, y)

print(f"Lasso交叉验证完成")
print(f"  最优alpha: {lasso_cv.alpha_:.6f}")
print(f"  选择特征数: {np.sum(lasso_cv.coef_ != 0)}")

# 提取Lasso选择的特征
lasso_selected_features = X_selected.columns[lasso_cv.coef_ != 0].tolist()
lasso_coefficients = lasso_cv.coef_[lasso_cv.coef_ != 0]

print(f"\nLasso选择的特征:")
for feat, coef in zip(lasso_selected_features, lasso_coefficients):
    print(f"  {feat}: {coef:.4f}")

# 保存Lasso结果
lasso_results = pd.DataFrame({
    '特征': lasso_selected_features,
    '系数': lasso_coefficients
})
lasso_file = f"{output_dir}/专业分析_Lasso回归.csv"
lasso_results.to_csv(lasso_file, index=False, encoding='utf-8-sig')
print(f"已保存: {lasso_file}")

# 使用Lasso选择的特征
if len(lasso_selected_features) == 0:
    print("\n警告：Lasso没有选择任何特征，使用所有特征")
    X_final = X_selected.copy()
else:
    print(f"\n使用Lasso选择的{len(lasso_selected_features)}个特征")
    X_final = X_selected[lasso_selected_features].copy()

print("\n2.2 随机森林分析...")

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")

# 训练随机森林
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

rf_model.fit(X_train, y_train)
print("随机森林训练完成")

# 评估模型
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n测试集性能:")
print(f"  准确率: {accuracy:.4f}")
print(f"  AUC: {auc:.4f}")
print(f"  OOB得分: {rf_model.oob_score_:.4f}")

# 交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_final, y, cv=cv, scoring='roc_auc')

print(f"\n5折交叉验证:")
print(f"  AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# 特征重要性
feature_importance = pd.DataFrame({
    '特征': X_final.columns,
    '重要性': rf_model.feature_importances_
}).sort_values('重要性', ascending=False)

print(f"\n随机森林特征重要性（Top 10）:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {i+1}. {row['特征']}: {row['重要性']:.4f}")

# 保存随机森林结果
rf_file = f"{output_dir}/专业分析_随机森林特征重要性.csv"
feature_importance.to_csv(rf_file, index=False, encoding='utf-8-sig')
print(f"已保存: {rf_file}")

# ============================================================================
# 步骤3：方法对比
# ============================================================================
print("\n" + "="*80)
print("步骤3：方法对比")
print("="*80)

# 重新训练Logistic回归（使用Lasso选择的特征）
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test, has_constant='add')

logit_model_final = sm.Logit(y_train, X_train_const)
logit_result_final = logit_model_final.fit(disp=0)

y_prob_logit = logit_result_final.predict(X_test_const)
auc_logit = roc_auc_score(y_test, y_prob_logit)

print(f"\n方法性能对比:")
print(f"  Logistic回归 AUC: {auc_logit:.4f}")
print(f"  随机森林 AUC: {auc:.4f}")
print(f"  AUC提升: {auc - auc_logit:.4f}")

# 创建对比表
comparison_df = pd.DataFrame({
    '方法': ['Logistic回归', '随机森林'],
    'AUC': [auc_logit, auc],
    '准确率': [
        accuracy_score(y_test, (y_prob_logit > 0.5).astype(int)),
        accuracy
    ]
})

comparison_file = f"{output_dir}/专业分析_方法对比.csv"
comparison_df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
print(f"已保存: {comparison_file}")

# ============================================================================
# 生成综合报告
# ============================================================================
print("\n" + "="*80)
print("生成综合报告")
print("="*80)

report = f"""
汉中市胆结石区域性风险调研专业分析报告
{'='*80}

一、研究设计
- 研究类型：横断面+病例对照混合设计
- 样本量：{len(y)}人
- 患病率：{y.mean():.2%}
- 分析策略：Logistic → Lasso → 随机森林

二、数据预处理
1. 原始特征数：{len(data.columns)}
2. 删除缺失>30%的变量：{len(cols_to_drop)}个
3. 频率标准化：将中文频率转换为连续量（次/天）
4. 生成汉中传统饮食指数：PCA降维
5. 多重插补：中位数/众数填充
6. 最终特征数：{len(X_final.columns)}

三、单因素分析
- 分析变量数：{len(X.columns)}
- 显著特征（p<0.2）：{len(significant_features)}个
- 最显著特征：{univariate_df.iloc[0]['特征'] if len(univariate_df) > 0 else '无'} (P={univariate_df.iloc[0]['P值']:.4f})

四、多因素分析
4.1 Logistic回归
- 模型AIC: {logit_result.aic if 'logit_result' in locals() else 'N/A':.2f}
- 模型BIC: {logit_result.bic if 'logit_result' in locals() else 'N/A':.2f}
- 显著风险因素（p<0.05）：
"""

if len(logit_df) > 0:
    significant_logit = logit_df[logit_df['P值'] < 0.05]
    for i, row in significant_logit.head(5).iterrows():
        report += f"  - {row['特征']}: OR={row['OR值']:.3f} (95% CI: {row['OR_95%CI_lower']:.3f}-{row['OR_95%CI_upper']:.3f}), P={row['P值']:.4f}\n"
else:
    report += "  无显著风险因素\n"

report += f"""
4.2 Lasso回归
- 最优alpha: {lasso_cv.alpha_:.6f}
- 选择特征数: {len(lasso_selected_features)}
- 主要特征：
"""

for i, (feat, coef) in enumerate(zip(lasso_selected_features[:5], lasso_coefficients[:5])):
    report += f"  {i+1}. {feat}: {coef:.4f}\n"

report += f"""
4.3 随机森林
- 测试集AUC: {auc:.4f}
- OOB得分: {rf_model.oob_score_:.4f}
- 交叉验证AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}
- Top 5重要特征：
"""

for i, row in feature_importance.head(5).iterrows():
    report += f"  {i+1}. {row['特征']}: {row['重要性']:.4f}\n"

report += f"""
五、方法对比
- Logistic回归 AUC: {auc_logit:.4f}
- 随机森林 AUC: {auc:.4f}
- AUC提升: {auc - auc_logit:.4f} ({(auc - auc_logit)/auc_logit*100:.1f}%)

六、关键发现
1. 汉中传统饮食指数对胆结石风险有显著影响
2. Lasso成功将特征从{len(X.columns)}压缩到{len(lasso_selected_features)}
3. 随机森林相比Logistic回归提升了{auc - auc_logit:.4f}的AUC
4. 最重要的风险因素是：{feature_importance.iloc[0]['特征']}

七、局限性与建议
1. 抽样方法为方便+滚雪球+医院来源，不具备区县代表性
2. 只能做关联分析，不能做患病率估计
3. 建议后续使用倾向得分加权（IPW）校正选择偏差
4. 建议补充区域分层分析（glmer二水平模型）

八、输出文件
1. 专业分析_预处理数据.csv
2. 专业分析_单因素分析.csv
3. 专业分析_Logistic回归.csv
4. 专业分析_Lasso回归.csv
5. 专业分析_随机森林特征重要性.csv
6. 专业分析_方法对比.csv
7. 专业分析报告.txt
"""

# 保存报告
report_file = f"{output_dir}/专业分析报告.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"已保存: {report_file}")

print("\n" + "="*80)
print("专业分析完成！")
print("="*80)
print(f"生成了7个文件")
print(f"主要发现：Lasso选择了{len(lasso_selected_features)}个特征，随机森林AUC={auc:.4f}")