import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel('原始数据.xlsx', sheet_name='Sheet1', header=0, skiprows=[1], usecols=range(67))

# 测试频率解析函数
def parse_hanzhong_frequency(freq_str):
    if isinstance(freq_str, str):
        if '<=3次/月' in freq_str or '3次/月' in freq_str:
            return 0
        elif '3-7次/月' in freq_str:
            return 1
        elif '>=7次/月' in freq_str or '7次/月' in freq_str:
            return 2
    return np.nan

print('测试parse_hanzhong_frequency函数:')
test_values = ['<=3次/月', '3-7次/月', '>=7次/月']
for val in test_values:
    result = parse_hanzhong_frequency(val)
    print(f'{val} -> {result}')

print('\n实际应用 - 汉中米皮:')
print(df.iloc[:, 18].apply(parse_hanzhong_frequency).value_counts())

print('\n实际应用 - 菜豆腐:')
print(df.iloc[:, 19].apply(parse_hanzhong_frequency).value_counts())

print('\n实际应用 - 浆水面:')
print(df.iloc[:, 20].apply(parse_hanzhong_frequency).value_counts())