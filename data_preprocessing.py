"""
汉中市胆结石风险调研数据预处理脚本
功能：数据清洗、编码、标准化
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = None
        self.cleaned_data = None
        self.encoded_data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_info = []
        self.target_column = None

    def load_data(self):
        """加载原始数据"""
        print("正在加载数据...")
        self.raw_data = pd.read_excel(self.file_path)
        print(f"数据加载完成: {self.raw_data.shape}")
        return self.raw_data

    def clean_data(self):
        """数据清洗"""
        print("\n正在进行数据清洗...")

        # 复制原始数据
        df = self.raw_data.copy()

        # 1. 删除缺失值过多的行（超过50%列缺失）
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold)
        print(f"删除缺失值过多的行后: {df.shape}")

        # 2. 处理缺失值
        for column in df.columns:
            if df[column].dtype in ['object', 'category']:
                # 分类型变量用众数填充
                mode_value = df[column].mode()
                if len(mode_value) > 0:
                    df[column].fillna(mode_value[0], inplace=True)
            else:
                # 数值型变量用均值填充
                df[column].fillna(df[column].mean(), inplace=True)

        # 3. 删除序号列（如果是数值型且唯一值数量等于行数）
        if '序号' in df.columns:
            df.drop('序号', axis=1, inplace=True)
            print("已删除序号列")

        self.cleaned_data = df
        print(f"数据清洗完成: {df.shape}")
        return df

    def identify_target_variable(self):
        """识别目标变量"""
        print("\n正在识别目标变量...")

        # 常见的目标变量名称
        possible_targets = [
            '1.您是否患有胆结石？',
            '胆结石发病状态',
            '是否患有胆结石',
            '胆结石'
        ]

        self.target_column = None
        for target in possible_targets:
            if target in self.cleaned_data.columns:
                self.target_column = target
                print(f"识别到目标变量: {target}")
                break

        # 如果没有找到，使用第一个列作为目标变量（通常是胆结石发病状态）
        if self.target_column is None and len(self.cleaned_data.columns) > 0:
            self.target_column = self.cleaned_data.columns[0]
            print(f"使用第一列作为目标变量: {self.target_column}")

        return self.target_column

    def encode_categorical_variables(self):
        """编码分类变量"""
        print("\n正在进行变量编码...")

        df = self.cleaned_data.copy()

        # 记录每个变量的信息
        self.feature_info = []

        for column in df.columns:
            if df[column].dtype in ['object', 'category']:
                # 分类型变量
                unique_values = df[column].unique()
                self.label_encoders[column] = LabelEncoder()

                # 处理缺失值和异常值
                df[column] = df[column].astype(str)
                df[column] = self.label_encoders[column].fit_transform(df[column])

                self.feature_info.append({
                    '列名': column,
                    '类型': '分类',
                    '唯一值数量': len(unique_values),
                    '编码方式': 'LabelEncoder',
                    '原始值': list(unique_values)[:10] if len(unique_values) <= 10 else f"{len(unique_values)}个值"
                })
            else:
                # 数值型变量
                self.feature_info.append({
                    '列名': column,
                    '类型': '数值',
                    '唯一值数量': df[column].nunique(),
                    '编码方式': '无需编码'
                })

        self.encoded_data = df
        print(f"变量编码完成: {df.shape}")
        return df

    def standardize_numeric_features(self):
        """标准化数值型特征"""
        print("\n正在进行特征标准化...")

        df = self.encoded_data.copy()

        # 识别数值型列（不包括目标变量）
        numeric_cols = []
        for item in self.feature_info:
            if item['类型'] == '数值' and item['列名'] != self.target_column:
                numeric_cols.append(item['列名'])

        if numeric_cols:
            # 对数值型列进行标准化
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            print(f"已标准化 {len(numeric_cols)} 个数值型特征")
        else:
            print("没有需要标准化的数值型特征")

        return df

    def create_preprocessed_dataset(self):
        """创建预处理后的数据集"""
        print("\n正在创建标准化数据集...")

        # 确保目标变量在最后一列
        if self.target_column and self.target_column in self.encoded_data.columns:
            target_col = self.encoded_data[self.target_column]
            feature_cols = [col for col in self.encoded_data.columns if col != self.target_column]
            df_final = pd.concat([self.encoded_data[feature_cols], target_col], axis=1)
        else:
            df_final = self.encoded_data.copy()

        return df_final

    def save_results(self, output_dir="."):
        """保存预处理结果"""
        print(f"\n正在保存结果到 {output_dir}...")

        # 1. 保存预处理后的数据
        preprocessed_file = f"{output_dir}/预处理数据.csv"
        self.create_preprocessed_dataset().to_csv(preprocessed_file, index=False, encoding='utf-8-sig')
        print(f"已保存预处理数据: {preprocessed_file}")

        # 2. 保存编码映射表
        encoding_mapping = {}
        for col, le in self.label_encoders.items():
            encoding_mapping[col] = {
                'classes': list(le.classes_),
                'encoded_values': list(range(len(le.classes_)))
            }

        mapping_file = f"{output_dir}/编码映射表.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(encoding_mapping, f, ensure_ascii=False, indent=2)
        print(f"已保存编码映射表: {mapping_file}")

        # 3. 保存特征信息
        feature_info_file = f"{output_dir}/特征信息.csv"
        pd.DataFrame(self.feature_info).to_csv(feature_info_file, index=False, encoding='utf-8-sig')
        print(f"已保存特征信息: {feature_info_file}")

        # 4. 保存数据质量报告
        quality_report = f"""
数据预处理报告
{'='*60}

原始数据形状: {self.raw_data.shape}
清洗后数据形状: {self.cleaned_data.shape}
编码后数据形状: {self.encoded_data.shape}

目标变量: {self.target_column if self.target_column else '未指定'}

特征统计:
- 总特征数: {len(self.feature_info)}
- 分类特征数: {len([f for f in self.feature_info if f['类型'] == '分类'])}
- 数值特征数: {len([f for f in self.feature_info if f['类型'] == '数值'])}

缺失值处理:
- 删除缺失值过多的行: {len(self.raw_data) - len(self.cleaned_data)} 行
- 填充缺失值: 已完成

变量编码:
- 使用LabelEncoder对分类变量进行编码
- 使用StandardScaler对数值变量进行标准化

输出文件:
1. 预处理数据.csv - 预处理后的数据集
2. 编码映射表.json - 变量编码对照表
3. 特征信息.csv - 特征详细信息
"""

        report_file = f"{output_dir}/预处理报告.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(quality_report)
        print(f"已保存预处理报告: {report_file}")

        print("\n所有结果保存完成！")

    def run_preprocessing(self, output_dir="."):
        """运行完整的预处理流程"""
        print("="*60)
        print("汉中市胆结石风险调研数据预处理")
        print("="*60)

        # 1. 加载数据
        self.load_data()

        # 2. 清洗数据
        self.clean_data()

        # 3. 识别目标变量
        self.identify_target_variable()

        # 4. 编码分类变量
        self.encode_categorical_variables()

        # 5. 标准化数值特征
        self.standardize_numeric_features()

        # 6. 保存结果
        self.save_results(output_dir)

        print("\n预处理流程完成！")
        return self.create_preprocessed_dataset()

if __name__ == "__main__":
    # 运行预处理
    preprocessor = DataPreprocessor(r"C:\Users\霍冠华\Documents\trae_projects\claude code\原始数据.xlsx")
    preprocessed_data = preprocessor.run_preprocessing(r"C:\Users\霍冠华\Documents\trae_projects\claude code")

    print(f"\n预处理后的数据形状: {preprocessed_data.shape}")
    print(f"数据预览:")
    print(preprocessed_data.head())