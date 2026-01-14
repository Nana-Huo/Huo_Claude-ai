# -*- coding: utf-8 -*-
"""
独立的多因素Logistic回归分析
包含完整的数据预处理、模型训练和结果输出
可以直接运行，不依赖其他文件
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class LogisticRegressionAnalysis:
    """独立的多因素Logistic回归分析类"""

    def __init__(self, file_path=None):
        self.file_path = file_path or r"C:\Users\霍冠华\Desktop\毕设\数据处理\处理后数据_20260104_144216.xlsx"
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_results = None
        self.model = None

    def load_data(self):
        """加载数据"""
        try:
            self.data = pd.read_excel(self.file_path)
            print(f"✓ 数据加载成功")
            print(f"  数据形状: {self.data.shape}")
            print(f"  列名: {self.data.columns.tolist()}")
            return True
        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            return False

    def preprocess_data(self, target_column):
        """数据预处理"""
        print("\n=== 数据预处理 ===")

        # 检查目标变量
        if target_column not in self.data.columns:
            print(f"✗ 错误: 目标变量 '{target_column}' 不存在")
            print(f"  可用列名: {self.data.columns.tolist()}")
            return False

        # 处理缺失值
        print("\n处理缺失值...")
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            print("  缺失值统计:")
            for col, count in missing_values[missing_values > 0].items():
                print(f"    {col}: {count}")

            # 删除缺失值过多的行
            threshold = len(self.data.columns) * 0.5
            self.data = self.data.dropna(thresh=threshold)

            # 填充剩余缺失值
            for column in self.data.columns:
                if self.data[column].dtype in ['object', 'category']:
                    mode_value = self.data[column].mode()
                    if len(mode_value) > 0:
                        self.data[column].fillna(mode_value[0], inplace=True)
                else:
                    self.data[column].fillna(self.data[column].mean(), inplace=True)
        else:
            print("  ✓ 无缺失值")

        # 编码分类变量
        print("\n编码分类变量...")
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            print(f"  发现分类变量: {categorical_columns.tolist()}")
            for column in categorical_columns:
                le = LabelEncoder()
                self.data[column] = le.fit_transform(self.data[column].astype(str))
        else:
            print("  ✓ 无分类变量需要编码")

        # 分离特征和目标变量
        self.y = self.data[target_column]
        self.X = self.data.drop(columns=[target_column])
        self.feature_names = self.X.columns.tolist()

        # 分割数据集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )

        # 标准化特征
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"\n✓ 数据预处理完成")
        print(f"  特征数量: {self.X.shape[1]}")
        print(f"  样本数量: {self.X.shape[0]}")
        print(f"  训练集: {self.X_train.shape[0]}")
        print(f"  测试集: {self.X_test.shape[0]}")
        print(f"  目标变量分布: {dict(self.y.value_counts())}")

        return True

    def univariate_analysis(self):
        """单因素Logistic回归分析"""
        print("\n=== 单因素Logistic回归分析 ===")

        univariate_results = []

        for i, column in enumerate(self.feature_names):
            X_single = sm.add_constant(self.X_train_scaled[:, i:i+1])

            try:
                model = sm.Logit(self.y_train, X_single)
                result = model.fit(disp=False)

                # 提取关键信息
                p_value = result.pvalues[1]
                odds_ratio = np.exp(result.params[1])
                ci_lower, ci_upper = np.exp(result.conf_int().iloc[1])

                univariate_results.append({
                    '特征': column,
                    '系数': result.params[1],
                    '标准误': result.bse[1],
                    'P值': p_value,
                    'OR值': odds_ratio,
                    '95% CI下限': ci_lower,
                    '95% CI上限': ci_upper,
                    '显著性': '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else ''
                })

            except Exception as e:
                print(f"特征 {column} 分析失败: {e}")

        # 创建结果表格
        uni_df = pd.DataFrame(univariate_results)
        if len(uni_df) > 0:
            uni_df = uni_df.sort_values('P值')

            print("\n单因素分析结果 (按P值排序):")
            print(uni_df.round(4))

            # 保存结果
            uni_df.to_csv('logistic_univariate_results.csv', index=False, encoding='utf-8-sig')
            print("\n已保存到: logistic_univariate_results.csv")

            # 可视化显著变量
            significant_vars = uni_df[uni_df['P值'] < 0.05]
            if len(significant_vars) > 0:
                plt.figure(figsize=(10, 6))
                sns.barplot(data=significant_vars.head(10), x='OR值', y='特征')
                plt.title('单因素分析显著变量 (P<0.05) - OR值', fontsize=14)
                plt.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='OR=1')
                plt.xlabel('OR值')
                plt.legend()
                plt.tight_layout()
                plt.savefig('logistic_univariate_or.png', dpi=300, bbox_inches='tight')
                plt.show()

        return uni_df

    def multivariate_analysis(self):
        """多因素Logistic回归分析"""
        print("\n=== 多因素Logistic回归分析 ===")

        X_train_scaled = sm.add_constant(self.X_train_scaled)
        X_test_scaled = sm.add_constant(self.X_test_scaled)

        try:
            # 构建多因素Logistic回归模型
            self.model = sm.Logit(self.y_train, X_train_scaled)
            self.model_results = self.model.fit(disp=False)

            # 打印详细结果
            print("\n模型摘要:")
            print(self.model_results.summary())

            # 提取显著变量
            p_values = self.model_results.pvalues[1:]  # 排除常数项
            significant_features = []
            significant_indices = []

            for i, p_val in enumerate(p_values):
                if p_val < 0.05:
                    significant_features.append(self.feature_names[i])
                    significant_indices.append(i)

            print(f"\n显著变量 (P<0.05): {len(significant_features)}个")
            for feature in significant_features:
                print(f"  - {feature}")

            # 预测
            y_pred_prob = self.model_results.predict(X_test_scaled)
            y_pred = (y_pred_prob > 0.5).astype(int)

            # 评估指标
            accuracy = accuracy_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_prob)

            print(f"\n模型性能:")
            print(f"- 准确率: {accuracy:.4f}")
            print(f"- AUC: {auc:.4f}")

            # 分类报告
            print("\n分类报告:")
            print(classification_report(self.y_test, y_pred))

            # 混淆矩阵
            cm = confusion_matrix(self.y_test, y_pred)
            print("\n混淆矩阵:")
            print(cm)

            # 绘制ROC曲线
            self._plot_roc_curve(self.y_test, y_pred_prob)

            # 保存预测结果
            predictions_df = pd.DataFrame({
                '实际值': self.y_test,
                '预测概率': y_pred_prob,
                '预测类别': y_pred
            })
            predictions_df.to_csv('logistic_predictions.csv', index=False, encoding='utf-8-sig')
            print("\n预测结果已保存到: logistic_predictions.csv")

            # 返回结果
            return {
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_prob,
                'confusion_matrix': cm,
                'significant_features': significant_features
            }

        except Exception as e:
            print(f"多因素分析失败: {e}")
            return None

    def feature_selection(self):
        """特征选择"""
        print("\n=== 特征选择 ===")

        # 方法1: 基于P值的向后剔除
        print("\n方法1: 基于P值的向后剔除")
        selected_features = list(range(self.X_train_scaled.shape[1]))

        while len(selected_features) > 1:
            X_current = sm.add_constant(self.X_train_scaled[:, selected_features])
            model = sm.Logit(self.y_train, X_current)
            result = model.fit(disp=False)

            p_values = result.pvalues[1:]  # 排除常数项
            max_p = p_values.max()

            if max_p > 0.1:  # 剔除P值>0.1的变量
                worst_feature_idx = p_values.argmax()
                removed_idx = selected_features.pop(worst_feature_idx)
                print(f"移除特征: {self.feature_names[removed_idx]} (P={max_p:.4f})")
            else:
                break

        print(f"\n最终选择的特征数量: {len(selected_features)}")
        print("选择的特征:", [self.feature_names[i] for i in selected_features])

        # 方法2: 使用sklearn的RFE进行特征选择
        print("\n方法2: 递归特征消除 (RFE)")
        estimator = LogisticRegression(max_iter=1000)
        rfe = RFE(estimator, n_features_to_select=min(10, len(self.feature_names)))
        rfe = rfe.fit(self.X_train_scaled, self.y_train)

        selected_features_rfe = [self.feature_names[i] for i, selected in enumerate(rfe.support_) if selected]
        print(f"RFE选择的特征: {selected_features_rfe}")

    def model_diagnostics(self):
        """模型诊断"""
        print("\n=== 模型诊断 ===")

        if self.model_results is None:
            print("模型未建立，跳过诊断")
            return

        X_test_scaled = sm.add_constant(self.X_test_scaled)
        y_pred_prob = self.model_results.predict(X_test_scaled)

        # 1. 残差分析
        print("\n1. 残差分析")
        residuals = self.model_results.resid_response
        fitted_values = self.model_results.fittedvalues

        plt.figure(figsize=(12, 4))

        # 残差vs拟合值
        plt.subplot(1, 2, 1)
        plt.scatter(fitted_values, residuals, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('拟合值')
        plt.ylabel('残差')
        plt.title('残差vs拟合值')

        # Q-Q图
        plt.subplot(1, 2, 2)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q图')

        plt.tight_layout()
        plt.savefig('logistic_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. 多重共线性检验
        print("\n2. 多重共线性检验 (VIF)")
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            # 计算VIF
            vif_data = []
            for i in range(self.X_train_scaled.shape[1]):
                try:
                    vif = variance_inflation_factor(self.X_train_scaled, i)
                    vif_data.append({
                        '特征': self.feature_names[i],
                        'VIF': vif,
                        '共线性': '严重' if vif > 10 else '中等' if vif > 5 else '轻微'
                    })
                except:
                    vif_data.append({
                        '特征': self.feature_names[i],
                        'VIF': np.inf,
                        '共线性': '严重'
                    })

            vif_df = pd.DataFrame(vif_data)
            vif_df = vif_df.sort_values('VIF', ascending=False)
            print(vif_df.round(3))

            vif_df.to_csv('logistic_vif.csv', index=False, encoding='utf-8-sig')
            print("\nVIF分析结果已保存到: logistic_vif.csv")
        except Exception as e:
            print(f"VIF计算失败: {e}")

        # 3. 预测概率分布
        print("\n3. 预测概率分布")
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.hist(y_pred_prob, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('预测概率')
        plt.ylabel('频数')
        plt.title('预测概率分布')

        plt.subplot(1, 2, 2)
        for label in [0, 1]:
            subset = y_pred_prob[self.y_test == label]
            if len(subset) > 0:
                plt.hist(subset, bins=20, alpha=0.5, label=f'类别 {label}')
        plt.xlabel('预测概率')
        plt.ylabel('频数')
        plt.title('按实际类别的预测概率分布')
        plt.legend()

        plt.tight_layout()
        plt.savefig('logistic_probability_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_roc_curve(self, y_true, y_pred_prob):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        auc = roc_auc_score(y_true, y_pred_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Logistic回归 (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
        plt.xlabel('假阳性率 (FPR)')
        plt.ylabel('真阳性率 (TPR)')
        plt.title('Logistic回归 - ROC曲线', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('logistic_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self, results):
        """生成分析报告"""
        print("\n=== 生成分析报告 ===")

        report_content = []
        report_content.append("# 多因素Logistic回归分析报告\n\n")
        report_content.append(f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_content.append(f"**数据文件**: {self.file_path}\n\n")

        # 数据概览
        report_content.append("## 1. 数据概览\n")
        report_content.append(f"- 原始数据形状: {self.data.shape}\n")
        report_content.append(f"- 特征数量: {len(self.feature_names)}\n")
        report_content.append(f"- 样本数量: {len(self.y)}\n")
        report_content.append(f"- 目标变量分布: {dict(self.y.value_counts())}\n\n")

        # 模型性能
        report_content.append("## 2. 模型性能\n")
        report_content.append(f"- 准确率: {results['accuracy']:.4f}\n")
        report_content.append(f"- AUC: {results['auc']:.4f}\n")
        report_content.append(f"- 显著特征数: {len(results['significant_features'])}\n\n")

        # 显著特征
        if results['significant_features']:
            report_content.append("## 3. 显著特征 (P<0.05)\n")
            for feature in results['significant_features']:
                report_content.append(f"- {feature}\n")
            report_content.append("\n")

        # 分类报告
        report_content.append("## 4. 分类报告\n")
        report_content.append("```\n")
        report_content.append(classification_report(self.y_test, results['predictions']))
        report_content.append("```\n\n")

        # 结论
        report_content.append("## 5. 结论\n")
        if results['accuracy'] > 0.8:
            report_content.append("模型性能优秀，可用于实际预测。\n")
        elif results['accuracy'] > 0.7:
            report_content.append("模型性能良好，建议进一步优化。\n")
        else:
            report_content.append("模型性能一般，需要改进特征选择或尝试其他方法。\n")

        # 保存报告
        with open('logistic_regression_report.md', 'w', encoding='utf-8') as f:
            f.writelines(report_content)

        print("✓ 分析报告已保存到: logistic_regression_report.md")

    def run_analysis(self, target_column):
        """运行完整的分析流程"""
        print("=" * 80)
        print("多因素Logistic回归分析")
        print("=" * 80)

        # 1. 加载数据
        if not self.load_data():
            return False

        # 2. 预处理数据
        if not self.preprocess_data(target_column):
            return False

        # 3. 单因素分析
        self.univariate_analysis()

        # 4. 多因素分析
        results = self.multivariate_analysis()
        if results is None:
            return False

        # 5. 特征选择
        self.feature_selection()

        # 6. 模型诊断
        self.model_diagnostics()

        # 7. 生成报告
        self.generate_report(results)

        print("\n" + "=" * 80)
        print("✓ Logistic回归分析完成！")
        print("=" * 80)
        print("\n生成的文件:")
        print("- logistic_univariate_results.csv - 单因素分析结果")
        print("- logistic_predictions.csv - 预测结果")
        print("- logistic_vif.csv - VIF多重共线性检验")
        print("- logistic_regression_report.md - 分析报告")
        print("- logistic_roc_curve.png - ROC曲线")
        print("- logistic_diagnostics.png - 模型诊断图")
        print("- logistic_probability_distribution.png - 预测概率分布")
        print("- logistic_univariate_or.png - 单因素OR值图")

        return True


def main():
    """主函数"""
    print("=" * 80)
    print("多因素Logistic回归分析工具")
    print("=" * 80)

    # 创建分析器
    analyzer = LogisticRegressionAnalysis()

    # 获取目标变量
    print("\n可用的列名:")
    if analyzer.load_data():
        print(analyzer.data.columns.tolist())

    target_column = input("\n请输入目标变量列名: ").strip()

    if not target_column:
        print("\n未指定目标变量，程序退出")
        return

    # 运行分析
    analyzer.run_analysis(target_column)


if __name__ == "__main__":
    main()