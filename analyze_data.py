import pandas as pd
import numpy as np

def analyze_excel_data(file_path):
    """分析Excel文件的数据结构"""
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)

        print("=" * 60)
        print("数据基本信息")
        print("=" * 60)
        print(f"数据形状: {df.shape} (行数: {df.shape[0]}, 列数: {df.shape[1]})")
        print()

        print("=" * 60)
        print("列名列表")
        print("=" * 60)
        for i, col in enumerate(df.columns):
            print(f"{i+1:2d}. {col}")
        print()

        print("=" * 60)
        print("前10行数据预览")
        print("=" * 60)
        print(df.head(10))
        print()

        print("=" * 60)
        print("数据类型")
        print("=" * 60)
        print(df.dtypes)
        print()

        print("=" * 60)
        print("缺失值统计")
        print("=" * 60)
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_info = pd.DataFrame({
            '缺失数量': missing_data,
            '缺失比例(%)': missing_percent
        })
        print(missing_info[missing_info['缺失数量'] > 0] if missing_info['缺失数量'].sum() > 0 else "没有缺失值")
        print()

        print("=" * 60)
        print("数值型列的描述性统计")
        print("=" * 60)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe())
        else:
            print("没有数值型列")
        print()

        print("=" * 60)
        print("字符型列的唯一值数量")
        print("=" * 60)
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            for col in object_cols:
                unique_count = df[col].nunique()
                print(f"{col}: {unique_count} 个唯一值")
                if unique_count <= 10:  # 如果唯一值较少，显示所有值
                    print(f"  值: {list(df[col].unique())}")
        else:
            print("没有字符型列")

        return df

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

if __name__ == "__main__":
    file_path = r"C:\Users\霍冠华\Documents\trae_projects\claude code\原始数据.xlsx"
    df = analyze_excel_data(file_path)