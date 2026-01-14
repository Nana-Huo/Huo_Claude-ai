import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# 读取数据
df = pd.read_excel('原始数据.xlsx', sheet_name='Sheet1', header=0, skiprows=[1], usecols=range(67))

# 目标变量
target = df.iloc[:, 1].apply(lambda x: 1 if x == '有' else 0)

# 汉中特色食品
features = ['汉中米皮', '菜豆腐', '浆水面', '火锅锅底']
cols = [18, 19, 20, 21]

print('汉中特色食品卡方检验结果：\n')

for feature, col in zip(features, cols):
    ct = pd.crosstab(df.iloc[:, col], target)
    chi2, p, dof, expected = chi2_contingency(ct)
    print(f'{feature}:')
    print(f'  卡方值 = {chi2:.3f}')
    print(f'  P值 = {p:.4f}')
    print(f'  显著性: {"是" if p < 0.05 else "否"}')
    print(f'  列联表:\n{ct}\n')