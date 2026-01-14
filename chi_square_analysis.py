"""
汉中市胆结石风险调研 - 卡方检验分析
功能：使用卡方检验筛选与胆结石发病相关的潜在风险因素
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
import warnings

warnings.filterwarnings('ignore')

class ChiSquareAnalysis:
    def __init__(self, data_path, target_column=None):
        self.data_path = data_path
        self.data = None
        self.target_column = target_column
        self.results = []
        self.significant_features = []

    def load_data(self):
        """加载预处理后的数据"""
        print("正在加载数据...")
        self.data = pd.read_csv(self.data_path, encoding='utf-8-sig')
        print(f"数据加载完成: {self.data.shape}")

        # 如果没有指定目标变量，使用最后一列
        if self.target_column is None:
            self.target_column = self.data.columns[-1]
            print(f"使用最后一列作为目标变量: {self.target_column}")

        return self.data

    def perform_chi_square_test(self, feature_column):
        """对单个特征执行卡方检验"""
        try:
            # 创建交叉表
            contingency_table = pd.crosstab(self.data[feature_column], self.data[self.target_column])

            # 检查交叉表的有效性
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                return None

            # 执行卡方检验
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            # 如果期望频数太低，使用Fisher精确检验
            if (expected < 5).sum() / expected.size > 0.2:
                # 对于2x2表格，使用Fisher精确检验
                if contingency_table.shape == (2, 2):
                    odds_ratio, p_value = fisher_exact(contingency_table)
                    test_method = "Fisher精确检验"
                else:
                    test_method = "卡方检验（期望频数较低）"
            else:
                test_method = "卡方检验"

            # 计算效应量（Cramer's V）
            n = contingency_table.sum().sum()
            phi = np.sqrt(chi2 / n)
            k = min(contingency_table.shape) - 1
            cramers_v = phi / np.sqrt(k - 1)

            return {
                'feature': feature_column,
                'test_method': test_method,
                'chi2': chi2,
                'p_value': p_value,
                'dof': dof,
                'cramers_v': cramers_v,
                'significance': '显著' if p_value < 0.05 else '不显著',
                'effect_size': '大' if cramers_v > 0.3 else ('中' if cramers_v > 0.1 else '小')
            }

        except Exception as e:
            print(f"特征 {feature_column} 分析失败: {e}")
            return None

    def run_analysis(self):
        """对所有特征执行卡方检验"""
        print("\n" + "="*60)
        print("开始卡方检验分析")
        print("="*60)

        # 获取所有特征列（排除目标变量）
        feature_columns = [col for col in self.data.columns if col != self.target_column]

        print(f"待分析特征数量: {len(feature_columns)}")
        print(f"目标变量: {self.target_column}")
        print()

        # 对每个特征执行卡方检验
        for i, feature in enumerate(feature_columns):
            if (i + 1) % 20 == 0:
                print(f"已分析 {i+1}/{len(feature_columns)} 个特征...")

            result = self.perform_chi_square_test(feature)
            if result:
                self.results.append(result)

        print(f"\n分析完成！共分析 {len(self.results)} 个特征")

        # 筛选显著特征
        self.significant_features = [r for r in self.results if r['p_value'] < 0.05]
        print(f"其中显著特征数量: {len(self.significant_features)} (P < 0.05)")

        return self.results

    def create_results_dataframe(self):
        """创建结果数据框"""
        print(f"调试: results长度 = {len(self.results)}")
        if len(self.results) > 0:
            print(f"调试: 第一个结果的键 = {self.results[0].keys()}")

        df = pd.DataFrame(self.results)

        # 按P值排序
        df = df.sort_values('p_value')

        # 添加排名
        df['rank'] = range(1, len(df) + 1)

        # 重新排列列
        df = df[['rank', 'feature', 'test_method', 'chi2', 'p_value', 'dof',
                 'cramers_v', 'effect_size', 'significance']]

        # 重命名列为中文
        df.columns = ['排名', '特征', '检验方法', '卡方值', 'P值', '自由度',
                     'Cramer\'s V', '效应量', '显著性']

        return df

    def save_results(self, output_dir="."):
        """保存分析结果"""
        print(f"\n正在保存结果到 {output_dir}...")

        # 1. 保存所有特征的分析结果
        all_results_df = self.create_results_dataframe()
        all_results_file = f"{output_dir}/卡方检验结果_所有特征.csv"
        all_results_df.to_csv(all_results_file, index=False, encoding='utf-8-sig')
        print(f"已保存所有特征结果: {all_results_file}")

        # 2. 保存显著特征的结果
        if len(self.significant_features) > 0:
            significant_df = pd.DataFrame(self.significant_features)
            significant_df = significant_df.sort_values('p_value')
            significant_df['rank'] = range(1, len(significant_df) + 1)
            significant_df = significant_df[['rank', 'feature', 'test_method', 'chi2', 'p_value',
                                            'dof', 'cramers_v', 'effect_size', 'significance']]
            significant_df.columns = ['排名', '特征', '检验方法', '卡方值', 'P值',
                                      '自由度', 'Cramer\'s V', '效应量', '显著性']

            significant_file = f"{output_dir}/卡方检验结果_显著特征.csv"
            significant_df.to_csv(significant_file, index=False, encoding='utf-8-sig')
            print(f"已保存显著特征结果: {significant_file}")

            # 3. 生成分析报告
            report = f"""
卡方检验分析报告
{'='*60}

数据信息:
- 数据集形状: {self.data.shape}
- 目标变量: {self.target_column}
- 分析特征数: {len(self.results)}
- 显著特征数: {len(self.significant_features)}
- 显著性水平: P < 0.05

统计摘要:
- 最小P值: {min([r['p_value'] for r in self.results]):.4f}
- 最大P值: {max([r['p_value'] for r in self.results]):.4f}
- 平均P值: {np.mean([r['p_value'] for r in self.results]):.4f}

显著特征列表 (前20个):
{'-'*60}
"""

            for i, feature in enumerate(self.significant_features[:20]):
                report += f"{i+1}. {feature['feature']}\n"
                report += f"   P值: {feature['p_value']:.4f}, 卡方值: {feature['chi2']:.2f}\n"
                report += f"   Cramer's V: {feature['cramers_v']:.4f}, 效应量: {feature['effect_size']}\n\n"

            if len(self.significant_features) > 20:
                report += f"... 还有 {len(self.significant_features) - 20} 个显著特征\n"

            report += f"""
效应量分布:
- 大效应量 (V > 0.3): {len([r for r in self.significant_features if r['effect_size'] == '大'])} 个
- 中等效应量 (0.1 < V <= 0.3): {len([r for r in self.significant_features if r['effect_size'] == '中'])} 个
- 小效应量 (V <= 0.1): {len([r for r in self.significant_features if r['effect_size'] == '小'])} 个

结论:
通过卡方检验，共识别出 {len(self.significant_features)} 个与胆结石发病显著相关的风险因素。
这些特征将在后续的多因素Logistic回归分析中进一步验证。

输出文件:
1. 卡方检验结果_所有特征.csv - 所有特征的分析结果
2. 卡方检验结果_显著特征.csv - 显著特征的分析结果
"""

            report_file = f"{output_dir}/卡方检验分析报告.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"已保存分析报告: {report_file}")

        else:
            print("警告：没有发现显著特征")

        print("\n所有结果保存完成！")

    def run_complete_analysis(self, output_dir="."):
        """运行完整的卡方检验分析流程"""
        print("="*60)
        print("汉中市胆结石风险调研 - 卡方检验分析")
        print("="*60)

        # 1. 加载数据
        self.load_data()

        # 2. 执行分析
        self.run_analysis()

        # 3. 保存结果
        self.save_results(output_dir)

        # 4. 显示摘要
        print("\n" + "="*60)
        print("分析摘要")
        print("="*60)
        print(f"总特征数: {len(self.results)}")
        print(f"显著特征数: {len(self.significant_features)}")

        if len(self.significant_features) > 0:
            print(f"\n前10个最显著的特征:")
            for i, feature in enumerate(self.significant_features[:10]):
                print(f"  {i+1}. {feature['feature']} (P={feature['p_value']:.4f}, V={feature['cramers_v']:.4f})")

        print("\n分析完成！")
        return self.results

if __name__ == "__main__":
    # 运行卡方检验分析
    analyzer = ChiSquareAnalysis(
        data_path=r"C:\Users\霍冠华\Documents\trae_projects\claude code\预处理数据.csv",
        target_column=None  # 使用最后一列作为目标变量
    )

    results = analyzer.run_complete_analysis(r"C:\Users\霍冠华\Documents\trae_projects\claude code")