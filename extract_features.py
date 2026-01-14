import pandas as pd

# 读取卡方检验结果
df = pd.read_csv(r'C:\Users\霍冠华\Documents\trae_projects\claude code\卡方检验结果_显著特征.csv', encoding='utf-8-sig')

# 提取特征列表
features = df['特征'].tolist()

# 保存到文本文件
with open(r'C:\Users\霍冠华\Documents\trae_projects\claude code\显著特征列表.txt', 'w', encoding='utf-8') as f:
    for feature in features:
        f.write(f"{feature}\n")

print(f"已提取 {len(features)} 个显著特征")
print("特征列表已保存到: 显著特征列表.txt")