import pandas as pd

# 读取数据
df = pd.read_excel('原始数据.xlsx', sheet_name='Sheet1', header=0, skiprows=[1], usecols=range(67))

# 获取第一个值
val = df.iloc[:, 18].iloc[0]

print('第一个值:', repr(val))
print('类型:', type(val))
print('长度:', len(val))
print('字符:', [c for c in val])
print('包含<=:', '<=' in val)
print('包含≤:', '≤' in val)
print('包含3次:', '3次' in val)
print('包含/月:', '/月' in val)

# 测试不同的匹配方式
print('\n测试匹配:')
print('<=3次/月 in val:', '<=3次/月' in val)
print('≤3次/月 in val:', '≤3次/月' in val)
print('3次/月 in val:', '3次/月' in val)
print('/月 in val:', '/月' in val)

# 显示前10个值
print('\n前10个值:')
for i in range(10):
    val = df.iloc[:, 18].iloc[i]
    print(f'{i}: {repr(val)}')