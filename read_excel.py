import pandas as pd

file_path = r'C:\Users\霍冠华\Documents\trae_projects\claude code\运价.xlsx'

try:
    # 使用openpyxl引擎读取.xlsx文件
    print("正在读取运价.xlsx文件...")
    df = pd.read_excel(file_path, engine='openpyxl')
    print(f"\n数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
    print("\n列名:")
    print(df.columns.tolist())
    print("\n数据类型:")
    print(df.dtypes.to_string())
    print("\n" + "="*80)
    print("\n前20行数据:")
    print(df.head(20).to_string())
    print("\n" + "="*80)
    print("\n数据统计信息:")
    print(df.describe(include='all').to_string())

except Exception as e:
    print(f"读取文件时出错: {e}")
    import traceback
    traceback.print_exc()