"""
汉中市胆结石风险调研 - 多因素Logistic回归分析
功能：识别独立风险因素，计算OR值和置信区间
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings('ignore')

class LogisticRegressionAnalysis:
    def __init__(self, data_path, target_column=None):
        self.data_path = data_path
        self.data = None
        self.target_column = target_column
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
        self.model = None
        self.results = None
        self.significant_features = []

    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        try:
            self.data = pd.read_csv(self.data_path, encoding='utf-8-sig')
            print(f"数据加载成功: {self.data.shape}")

            # 如果没有指定目标变量，使用最后一列
            if self.target_column is None:
                self.target_column = self.data.columns[-1]
                print(f"使用最后一列作为目标变量: {self.target_column}")

            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def prepare_data(self):
        """准备数据"""
        print("\n正在准备数据...")

        # 分离特征和目标变量
        self.y = self.data[self.target_column]
        self.X = self.data.drop(columns=[self.target_column])
        self.feature_names = self.X.columns.tolist()

        print(f"特征数量: {len(self.feature_names)}")
        print(f"目标变量: {self.target_column}")

        # 分割数据集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )

        print(f"训练集: {self.X_train.shape}")
        print(f"测试集: {self.X_test.shape}")

        # 标准化特征
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

        return True

    def fit_model(self):
        """拟合Logistic回归模型"""
        print("\n正在拟合Logistic回归模型...")

        # 使用statsmodels进行详细分析
        X_train_sm = sm.add_constant(self.X_train_scaled)
        logit_model = sm.Logit(self.y_train, X_train_sm)

        try:
            self.results = logit_model.fit(disp=False)
            print("模型拟合成功")
            return True
        except Exception as e:
            print(f"模型拟合失败: {e}")
            return False

    def analyze_results(self):
        """分析结果"""
        print("\n正在分析结果...")

        # 获取结果摘要
        summary = self.results.summary()

        # 提取关键信息
        params = self.results.params
        pvalues = self.results.pvalues
        conf_int = self.results.conf_int()

        # 创建结果数据框
        results_df = pd.DataFrame({
            '特征': ['截距'] + self.feature_names,
            '系数': params.values,
            '标准误': self.results.bse.values,
            'Z值': self.results.tvalues.values,
            'P值': pvalues.values,
            'OR值': np.exp(params.values),
            'OR_95%_CI_下限': np.exp(conf_int[0].values),
            'OR_95%_CI_上限': np.exp(conf_int[1].values)
        })

        # 添加显著性标记
        results_df['显著性'] = results_df['P值'].apply(
            lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else ''))
        )

        # 筛选显著特征（不包括截距）
        self.significant_features = results_df[
            (results_df['P值'] < 0.05) & (results_df['特征'] != '截距')
        ]

        print(f"显著特征数量: {len(self.significant_features)}")

        return results_df

    def check_multicollinearity(self):
        """检查多重共线性"""
        print("\n正在检查多重共线性...")

        # 计算VIF
        vif_data = []
        for i in range(self.X_train_scaled.shape[1]):
            vif = variance_inflation_factor(self.X_train_scaled, i)
            vif_data.append({
                '特征': self.feature_names[i],
                'VIF': vif,
                '共线性': '严重' if vif > 10 else ('中等' if vif > 5 else '轻微')
            })

        vif_df = pd.DataFrame(vif_data)

        # 找出有严重共线性的特征
        high_vif_features = vif_df[vif_df['VIF'] > 10]
        if len(high_vif_features) > 0:
            print(f"警告: 发现 {len(high_vif_features)} 个特征存在严重多重共线性")
            print(high_vif_features)
        else:
            print("未发现严重的多重共线性")

        return vif_df

    def evaluate_model(self):
        """评估模型性能"""
        print("\n正在评估模型性能...")

        # 在测试集上进行预测
        X_test_sm = sm.add_constant(self.X_test_scaled)
        y_pred_prob = self.results.predict(X_test_sm)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # 计算性能指标
        accuracy = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_prob)

        print(f"准确率: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")

        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\n混淆矩阵:")
        print(cm)

        return {
            'accuracy': accuracy,
            'auc': auc,
            'confusion_matrix': cm
        }

    def save_results(self, output_dir="."):
        """保存结果"""
        print(f"\n正在保存结果到 {output_dir}...")

        # 1. 保存模型结果
        results_df = self.analyze_results()
        results_file = f"{output_dir}/Logistic回归结果.csv"
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"已保存Logistic回归结果: {results_file}")

        # 2. 保存显著特征
        if len(self.significant_features) > 0:
            significant_file = f"{output_dir}/Logistic回归_显著特征.csv"
            self.significant_features.to_csv(significant_file, index=False, encoding='utf-8-sig')
            print(f"已保存显著特征: {significant_file}")

        # 3. 保存VIF结果
        vif_df = self.check_multicollinearity()
        vif_file = f"{output_dir}/多重共线性检验_VIF.csv"
        vif_df.to_csv(vif_file, index=False, encoding='utf-8-sig')
        print(f"已保存VIF结果: {vif_file}")

        # 4. 生成分析报告
        report = f"""
多因素Logistic回归分析报告
{'='*60}

数据信息:
- 数据集形状: {self.data.shape}
- 目标变量: {self.target_column}
- 特征数量: {len(self.feature_names)}
- 训练集大小: {self.X_train.shape[0]}
- 测试集大小: {self.X_test.shape[0]}

模型性能:
"""

        metrics = self.evaluate_model()
        report += f"- 准确率: {metrics['accuracy']:.4f}\n"
        report += f"- AUC: {metrics['auc']:.4f}\n"

        if len(self.significant_features) > 0:
            report += f"\n显著风险因素 (P < 0.05):\n"
            report += f"{'-'*60}\n"
            for i, row in self.significant_features.iterrows():
                report += f"{row['特征']}\n"
                report += f"  OR值: {row['OR值']:.4f} (95% CI: {row['OR_95%_CI_下限']:.4f}-{row['OR_95%_CI_上限']:.4f})\n"
                report += f"  P值: {row['P值']:.4f} {row['显著性']}\n\n"

        report += f"""
结论:
通过多因素Logistic回归分析，共识别出 {len(self.significant_features)} 个与胆结石发病显著相关的独立风险因素。
这些因素在控制其他变量后仍然显著，表明它们是胆结石发病的独立风险因素。

输出文件:
1. Logistic回归结果.csv - 所有特征的回归结果
2. Logistic回归_显著特征.csv - 显著特征的回归结果
3. 多重共线性检验_VIF.csv - VIF检验结果
"""

        report_file = f"{output_dir}/Logistic回归分析报告.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"已保存分析报告: {report_file}")

        print("\n所有结果保存完成！")

    def run_complete_analysis(self, output_dir="."):
        """运行完整的分析流程"""
        print("="*60)
        print("汉中市胆结石风险调研 - 多因素Logistic回归分析")
        print("="*60)

        # 1. 加载数据
        if not self.load_data():
            return None

        # 2. 准备数据
        if not self.prepare_data():
            return None

        # 3. 拟合模型
        if not self.fit_model():
            return None

        # 4. 分析结果
        results_df = self.analyze_results()

        # 5. 检查多重共线性
        self.check_multicollinearity()

        # 6. 评估模型
        self.evaluate_model()

        # 7. 保存结果
        self.save_results(output_dir)

        # 8. 显示摘要
        print("\n" + "="*60)
        print("分析摘要")
        print("="*60)
        print(f"总特征数: {len(self.feature_names)}")
        print(f"显著特征数: {len(self.significant_features)}")

        if len(self.significant_features) > 0:
            print(f"\n显著风险因素:")
            for i, row in self.significant_features.iterrows():
                print(f"  {row['特征']}: OR={row['OR值']:.4f}, P={row['P值']:.4f}")

        print("\n分析完成！")
        return results_df

if __name__ == "__main__":
    # 运行Logistic回归分析
    analyzer = LogisticRegressionAnalysis(
        data_path=r"C:\Users\霍冠华\Documents\trae_projects\claude code\预处理数据.csv",
        target_column=None  # 使用最后一列作为目标变量
    )

    results = analyzer.run_complete_analysis(r"C:\Users\霍冠华\Documents\trae_projects\claude code")