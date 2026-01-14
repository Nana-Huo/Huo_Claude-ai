import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel('原始数据.xlsx', sheet_name='Sheet1', header=0, skiprows=[1], usecols=range(67))

# 频率解析函数
def parse_hanzhong_frequency(freq_str):
    if isinstance(freq_str, str):
        if '<=3次/月' in freq_str or '≤3次/月' in freq_str:
            return 0
        elif '3-7次/月' in freq_str:
            return 1
        elif '>=7次/月' in freq_str or '≥7次/月' in freq_str:
            return 2
    return np.nan

# 提取汉中特色食品
df['汉中米皮'] = df.iloc[:, 18].apply(parse_hanzhong_frequency)
df['菜豆腐'] = df.iloc[:, 19].apply(parse_hanzhong_frequency)
df['浆水面'] = df.iloc[:, 20].apply(parse_hanzhong_frequency)

print('汉中米皮值分布:')
print(df['汉中米皮'].value_counts())

print('\n菜豆腐值分布:')
print(df['菜豆腐'].value_counts())

print('\n浆水面值分布:')
print(df['浆水面'].value_counts())

# 计算饮食模式_汉中特色
df['饮食模式_汉中特色'] = (df['汉中米皮'] + df['菜豆腐'] + df['浆水面']) / 3

print('\n饮食模式_汉中特色:')
print(df['饮食模式_汉中特色'].describe())

# 检查是否有NaN
print('\n饮食模式_汉中特色缺失值:')
print(df['饮食模式_汉中特色'].isna().sum())