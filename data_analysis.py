# -*- coding: utf-8 -*-
"""
数据分析脚本 - 毕业设计
使用三种分析方法：多因素Logistic回归、决策树、随机森林
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DataAnalyzer:
    def __init__(self, file_path):
        """
        初始化数据分析器

        参数:
        file_path: 数据文件路径
        """
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.results = {}

    def load_data(self):
        """加载数据"""
        try:
            self.data = pd.read_excel(self.file_path)
            print(f"数据加载成功，数据形状: {self.data.shape}")
            print("\n数据基本信息:")
            print(self.data.info())
            print("\n数据描述性统计:")
            print(self.data.describe())
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def preprocess_data(self, target_column):
        """
        数据预处理

        参数:
        target_column: 目标变量列名
        """
        if self.data is None:
            print("请先加载数据")
            return False

        print(f"\n开始数据预处理，目标变量: {target_column}")

        # 检查目标变量是否存在
        if target_column not in self.data.columns:
            print(f"错误: 目标变量 '{target_column}' 不存在于数据中")
            return False

        # 处理缺失值
        print("\n处理缺失值...")
        missing_values = self.data.isnull().sum()
        print("缺失值统计:")
        print(missing_values[missing_values > 0])

        # 删除包含过多缺失值的行（超过50%的列缺失）
        threshold = len(self.data.columns) * 0.5
        self.data = self.data.dropna(thresh=threshold)

        # 对数值型变量用均值填充，对分类型变量用众数填充
        for column in self.data.columns:
            if self.data[column].dtype in ['object', 'category']:
                # 分类型变量
                mode_value = self.data[column].mode()
                if len(mode_value) > 0:
                    self.data[column].fillna(mode_value[0], inplace=True)
            else:
                # 数值型变量
                self.data[column].fillna(self.data[column].mean(), inplace=True)

        # 编码分类变量
        print("\n编码分类变量...")
        label_encoders = {}
        for column in self.data.columns:
            if self.data[column].dtype in ['object', 'category']:
                le = LabelEncoder()
                self.data[column] = le.fit_transform(self.data[column].astype(str))
                label_encoders[column] = le

        # 分离特征和目标变量
        self.y = self.data[target_column]
        self.X = self.data.drop(columns=[target_column])

        print(f"特征数量: {self.X.shape[1]}")
        print(f"样本数量: {self.X.shape[0]}")
        print(f"目标变量分布: {dict(self.y.value_counts())}")

        # 分割训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )

        # 标准化特征
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("数据预处理完成")
        return True

    def logistic_regression_analysis(self):
        """多因素Logistic回归分析"""
        print("\n=== 多因素Logistic回归分析 ===")

        # 使用statsmodels进行详细的统计分析
        X_train_sm = sm.add_constant(self.X_train_scaled)

        try:
            # 构建Logistic回归模型
            logit_model = sm.Logit(self.y_train, X_train_sm)
            result = logit_model.fit(disp=False)

            print("\n模型摘要:")
            print(result.summary())

            # 预测
            X_test_sm = sm.add_constant(self.X_test_scaled)
            y_pred_prob = result.predict(X_test_sm)
            y_pred = (y_pred_prob > 0.5).astype(int)

            # 评估指标
            accuracy = accuracy_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_prob)

            print(f"\n准确率: {accuracy:.4f}")
            print(f"AUC: {auc:.4f}")

            # 分类报告
            print("\n分类报告:")
            print(classification_report(self.y_test, y_pred))

            # 混淆矩阵
            cm = confusion_matrix(self.y_test, y_pred)
            print("\n混淆矩阵:")
            print(cm)

            # 保存结果
            self.results['logistic_regression'] = {
                'model': result,
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_prob,
                'confusion_matrix': cm
            }

            # 绘制ROC曲线
            self._plot_roc_curve(self.y_test, y_pred_prob, 'Logistic回归')

            return True

        except Exception as e:
            print(f"Logistic回归分析失败: {e}")
            return False

    def decision_tree_analysis(self):
        """决策树分析"""
        print("\n=== 决策树分析 ===")

        try:
            # 网格搜索优化参数
            param_grid = {
                'max_depth': [3, 5, 7, 9, 11],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }

            dt = DecisionTreeClassifier(random_state=42)
            grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)

            best_dt = grid_search.best_estimator_
            print(f"最佳参数: {grid_search.best_params_}")

            # 预测
            y_pred = best_dt.predict(self.X_test)
            y_pred_prob = best_dt.predict_proba(self.X_test)[:, 1]

            # 评估指标
            accuracy = accuracy_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_prob)

            print(f"\n准确率: {accuracy:.4f}")
            print(f"AUC: {auc:.4f}")

            # 分类报告
            print("\n分类报告:")
            print(classification_report(self.y_test, y_pred))

            # 混淆矩阵
            cm = confusion_matrix(self.y_test, y_pred)
            print("\n混淆矩阵:")
            print(cm)

            # 特征重要性
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': best_dt.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\n特征重要性 (前10个):")
            print(feature_importance.head(10))

            # 保存结果
            self.results['decision_tree'] = {
                'model': best_dt,
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_prob,
                'confusion_matrix': cm,
                'feature_importance': feature_importance
            }

            # 绘制ROC曲线
            self._plot_roc_curve(self.y_test, y_pred_prob, '决策树')

            # 绘制决策树
            self._plot_decision_tree(best_dt)

            # 绘制特征重要性
            self._plot_feature_importance(feature_importance, '决策树特征重要性')

            return True

        except Exception as e:
            print(f"决策树分析失败: {e}")
            return False

    def random_forest_analysis(self):
        """随机森林分析"""
        print("\n=== 随机森林分析 ===")

        try:
            # 网格搜索优化参数
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)

            best_rf = grid_search.best_estimator_
            print(f"最佳参数: {grid_search.best_params_}")

            # 预测
            y_pred = best_rf.predict(self.X_test)
            y_pred_prob = best_rf.predict_proba(self.X_test)[:, 1]

            # 评估指标
            accuracy = accuracy_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_prob)

            print(f"\n准确率: {accuracy:.4f}")
            print(f"AUC: {auc:.4f}")

            # 分类报告
            print("\n分类报告:")
            print(classification_report(self.y_test, y_pred))

            # 混淆矩阵
            cm = confusion_matrix(self.y_test, y_pred)
            print("\n混淆矩阵:")
            print(cm)

            # 特征重要性
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': best_rf.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\n特征重要性 (前10个):")
            print(feature_importance.head(10))

            # 保存结果
            self.results['random_forest'] = {
                'model': best_rf,
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_prob,
                'confusion_matrix': cm,
                'feature_importance': feature_importance
            }

            # 绘制ROC曲线
            self._plot_roc_curve(self.y_test, y_pred_prob, '随机森林')

            # 绘制特征重要性
            self._plot_feature_importance(feature_importance, '随机森林特征重要性')

            return True

        except Exception as e:
            print(f"随机森林分析失败: {e}")
            return False

    def _plot_roc_curve(self, y_true, y_pred_prob, model_name):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        auc = roc_auc_score(y_true, y_pred_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
        plt.xlabel('假阳性率 (FPR)')
        plt.ylabel('真阳性率 (TPR)')
        plt.title(f'{model_name} - ROC曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{model_name}_ROC曲线.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_decision_tree(self, tree_model):
        """绘制决策树"""
        plt.figure(figsize=(20, 15))
        plot_tree(tree_model,
                 feature_names=self.X.columns,
                 class_names=[str(i) for i in sorted(self.y.unique())],
                 filled=True,
                 rounded=True,
                 max_depth=3)  # 只显示前3层以保持清晰
        plt.title('决策树结构 (前3层)', fontsize=16)
        plt.savefig('决策树结构.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_feature_importance(self, feature_importance, title):
        """绘制特征重要性"""
        top_features = feature_importance.head(15)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(title, fontsize=16)
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.tight_layout()
        plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def compare_models(self):
        """比较所有模型的性能"""
        print("\n=== 模型性能比较 ===")

        if not self.results:
            print("没有可比较的模型结果")
            return

        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                '模型': model_name,
                '准确率': result['accuracy'],
                'AUC': result['auc']
            })

        comparison_df = pd.DataFrame(comparison_data)
        print("\n模型性能比较表:")
        print(comparison_df.round(4))

        # 绘制比较图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 准确率比较
        sns.barplot(data=comparison_df, x='模型', y='准确率', ax=ax1)
        ax1.set_title('模型准确率比较')
        ax1.set_ylim(0, 1)

        # AUC比较
        sns.barplot(data=comparison_df, x='模型', y='AUC', ax=ax2)
        ax2.set_title('模型AUC比较')
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig('模型性能比较.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 保存比较结果
        comparison_df.to_csv('模型性能比较.csv', index=False, encoding='utf-8-sig')

    def generate_report(self):
        """生成分析报告"""
        print("\n=== 生成分析报告 ===")

        report = []
        report.append("# 数据分析报告\n")
        report.append(f"数据文件: {self.file_path}\n")
        report.append(f"分析时间: {pd.Timestamp.now()}\n")
        report.append(f"数据形状: {self.data.shape}\n")

        report.append("## 1. 数据概览\n")
        report.append(f"- 样本数量: {self.data.shape[0]}\n")
        report.append(f"- 特征数量: {self.data.shape[1] - 1}\n")
        report.append(f"- 目标变量分布: {dict(self.y.value_counts())}\n")

        report.append("## 2. 模型性能\n")
        for model_name, result in self.results.items():
            report.append(f"### {model_name}\n")
            report.append(f"- 准确率: {result['accuracy']:.4f}\n")
            report.append(f"- AUC: {result['auc']:.4f}\n")
            report.append("\n")

        # 找出最佳模型
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        report.append(f"## 3. 最佳模型\n")
        report.append(f"最佳模型: {best_model[0]}\n")
        report.append(f"准确率: {best_model[1]['accuracy']:.4f}\n")
        report.append(f"AUC: {best_model[1]['auc']:.4f}\n")

        # 保存报告
        with open('数据分析报告.md', 'w', encoding='utf-8') as f:
            f.writelines(report)

        print("分析报告已保存到 '数据分析报告.md'")

    def run_complete_analysis(self, target_column):
        """运行完整的数据分析流程"""
        print("开始完整数据分析流程...")

        # 加载数据
        if not self.load_data():
            return False

        # 预处理数据
        if not self.preprocess_data(target_column):
            return False

        # 运行三种分析方法
        self.logistic_regression_analysis()
        self.decision_tree_analysis()
        self.random_forest_analysis()

        # 比较模型
        self.compare_models()

        # 生成报告
        self.generate_report()

        print("\n数据分析完成！")
        return True


def main():
    """主函数"""
    # 数据文件路径
    file_path = r"C:\Users\霍冠华\Desktop\毕设\数据处理\处理后数据_20260104_144216.xlsx"

    # 创建分析器
    analyzer = DataAnalyzer(file_path)

    # 请用户指定目标变量列名
    print("请先查看数据列名，然后输入目标变量列名")

    # 这里需要根据实际数据情况修改
    target_column = input("请输入目标变量列名: ")

    if not target_column:
        print("未指定目标变量，使用默认假设...")
        # 这里可以根据实际数据设置一个默认值
        target_column = "target"  # 请根据实际数据修改

    # 运行完整分析
    analyzer.run_complete_analysis(target_column)


if __name__ == "__main__":
    main()