# -*- coding: utf-8 -*-
"""
独立的决策树分析
包含完整的数据预处理、模型训练和结果输出
可以直接运行，不依赖其他文件
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DecisionTreeAnalysis:
    """独立的决策树分析类"""

    def __init__(self, file_path=None):
        self.file_path = file_path or r"C:\Users\霍冠华\Desktop\毕设\数据处理\处理后数据_20260104_144216.xlsx"
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.model = None
        self.best_params = None
        self.feature_importance = None

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

        print(f"\n✓ 数据预处理完成")
        print(f"  特征数量: {self.X.shape[1]}")
        print(f"  样本数量: {self.X.shape[0]}")
        print(f"  训练集: {self.X_train.shape[0]}")
        print(f"  测试集: {self.X_test.shape[0]}")
        print(f"  目标变量分布: {dict(self.y.value_counts())}")

        return True

    def hyperparameter_tuning(self):
        """超参数调优"""
        print("\n=== 超参数调优 ===")

        # 定义参数网格
        param_grid = {
            'max_depth': [3, 5, 7, 9, 11, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', None]
        }

        # 创建决策树分类器
        dt = DecisionTreeClassifier(random_state=42)

        # 网格搜索
        print("正在进行网格搜索...")
        grid_search = GridSearchCV(
            dt, param_grid, cv=5,
            scoring='accuracy', n_jobs=-1,
            verbose=0
        )

        try:
            grid_search.fit(self.X_train, self.y_train)
        except Exception as e:
            print(f"网格搜索失败: {e}")
            # 使用默认参数
            self.model = DecisionTreeClassifier(random_state=42)
            self.best_params = self.model.get_params()
            return True

        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        print(f"\n最佳参数: {self.best_params}")
        print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")

        # 可视化参数调优结果
        self._plot_parameter_tuning_results(grid_search)

        return True

    def _plot_parameter_tuning_results(self, grid_search):
        """可视化参数调优结果"""
        cv_results = pd.DataFrame(grid_search.cv_results_)

        plt.figure(figsize=(15, 5))

        # 绘制不同max_depth的性能
        plt.subplot(1, 3, 1)
        depth_results = cv_results[cv_results['param_max_depth'].notna()]
        if len(depth_results) > 0:
            depth_performance = depth_results.groupby('param_max_depth')['mean_test_score'].mean()
            plt.plot(depth_performance.index, depth_performance.values, marker='o', markersize=8)
            plt.xlabel('最大深度')
            plt.ylabel('平均准确率')
            plt.title('不同树深度对性能的影响')
            plt.grid(True, alpha=0.3)

        # 绘制不同min_samples_split的性能
        plt.subplot(1, 3, 2)
        split_results = cv_results[cv_results['param_min_samples_split'].notna()]
        if len(split_results) > 0:
            split_performance = split_results.groupby('param_min_samples_split')['mean_test_score'].mean()
            plt.plot(split_performance.index, split_performance.values, marker='s', markersize=8)
            plt.xlabel('最小分割样本数')
            plt.ylabel('平均准确率')
            plt.title('不同最小分割样本数对性能的影响')
            plt.grid(True, alpha=0.3)

        # 绘制不同criterion的性能
        plt.subplot(1, 3, 3)
        criterion_results = cv_results[cv_results['param_criterion'].notna()]
        if len(criterion_results) > 0:
            criterion_performance = criterion_results.groupby('param_criterion')['mean_test_score'].mean()
            colors = ['skyblue', 'lightcoral']
            bars = plt.bar(criterion_performance.index, criterion_performance.values, color=colors[:len(criterion_performance)])
            plt.xlabel('分裂准则')
            plt.ylabel('平均准确率')
            plt.title('不同分裂准则对性能的影响')
            plt.grid(True, alpha=0.3)
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('decision_tree_parameter_tuning.png', dpi=300, bbox_inches='tight')
        plt.show()

    def train_model(self):
        """训练最终模型"""
        print("\n=== 训练最终模型 ===")

        # 使用最佳参数训练模型
        self.model.fit(self.X_train, self.y_train)

        # 预测
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        y_test_prob = self.model.predict_proba(self.X_test)[:, 1]

        # 评估训练集和测试集性能
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)

        print(f"训练集准确率: {train_accuracy:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")
        print(f"过拟合程度: {(train_accuracy - test_accuracy):.4f}")

        # 详细评估
        print(f"\n测试集性能:")
        print(f"准确率: {test_accuracy:.4f}")

        # 计算AUC
        auc = roc_auc_score(self.y_test, y_test_prob)
        print(f"AUC: {auc:.4f}")

        # 分类报告
        print("\n分类报告:")
        print(classification_report(self.y_test, y_test_pred))

        # 混淆矩阵
        cm = confusion_matrix(self.y_test, y_test_pred)
        print("\n混淆矩阵:")
        print(cm)

        # 绘制ROC曲线
        self._plot_roc_curve(self.y_test, y_test_prob)

        # 学习曲线分析
        self._plot_learning_curve()

        return {
            'accuracy': test_accuracy,
            'auc': auc,
            'predictions': y_test_pred,
            'probabilities': y_test_prob,
            'confusion_matrix': cm
        }

    def _plot_learning_curve(self):
        """绘制学习曲线"""
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X_train, self.y_train,
            cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
            random_state=42
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='训练集', linewidth=2)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

        plt.plot(train_sizes, test_mean, 'o-', color='red', label='验证集', linewidth=2)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')

        plt.xlabel('训练样本数')
        plt.ylabel('准确率')
        plt.title('决策树学习曲线', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('decision_tree_learning_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

    def feature_importance_analysis(self):
        """特征重要性分析"""
        print("\n=== 特征重要性分析 ===")

        # 计算特征重要性
        self.feature_importance = pd.DataFrame({
            '特征': self.feature_names,
            '重要性': self.model.feature_importances_
        }).sort_values('重要性', ascending=False)

        print("特征重要性排名:")
        print(self.feature_importance.round(4))

        # 可视化特征重要性
        plt.figure(figsize=(12, 8))

        # 条形图
        plt.subplot(2, 1, 1)
        top_features = self.feature_importance.head(15)
        bars = sns.barplot(data=top_features, x='重要性', y='特征')
        plt.title('决策树特征重要性 (前15个)', fontsize=14)
        plt.xlabel('重要性')
        # 添加数值标签
        for i, bar in enumerate(bars.patches):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center')
        plt.tight_layout()

        # 累积重要性
        plt.subplot(2, 1, 2)
        cumulative_importance = np.cumsum(self.feature_importance['重要性'])
        plt.plot(range(len(cumulative_importance)), cumulative_importance, marker='o', markersize=4)
        plt.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90%累积重要性')
        plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%累积重要性')
        plt.xlabel('特征数量')
        plt.ylabel('累积重要性')
        plt.title('特征累积重要性', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('decision_tree_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 保存特征重要性
        self.feature_importance.to_csv('decision_tree_feature_importance.csv', index=False, encoding='utf-8-sig')
        print("\n特征重要性已保存到: decision_tree_feature_importance.csv")

    def tree_visualization(self):
        """决策树可视化"""
        print("\n=== 决策树可视化 ===")

        # 绘制简化的决策树（只显示前几层）
        max_depth_simple = min(3, self.model.get_depth())
        plt.figure(figsize=(25, 15))
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=['0', '1'],  # 假设二分类
            filled=True,
            rounded=True,
            max_depth=max_depth_simple,
            fontsize=12,
            impurity=False,
            proportion=True
        )
        plt.title(f'决策树结构 (前{max_depth_simple}层)', fontsize=18)
        plt.savefig('decision_tree_structure.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 导出文本规则
        tree_rules = export_text(
            self.model,
            feature_names=self.feature_names,
            max_depth=5  # 只显示前5层
        )

        with open('decision_tree_rules.txt', 'w', encoding='utf-8') as f:
            f.write("决策树规则 (前5层):\n\n")
            f.write(tree_rules)

        print("决策树规则已保存到: decision_tree_rules.txt")

        # 打印树的信息
        print(f"\n树的信息:")
        print(f"- 树的深度: {self.model.get_depth()}")
        print(f"- 节点总数: {self.model.tree_.node_count}")
        print(f"- 叶子节点数: {self.model.get_n_leaves()}")

    def cross_validation(self):
        """交叉验证"""
        print("\n=== 交叉验证评估 ===")

        # 5折交叉验证
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train,
            cv=5, scoring='accuracy'
        )

        print(f"交叉验证得分: {cv_scores}")
        print(f"平均准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # 不同评分指标的交叉验证
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = {}

        print("\n各项指标交叉验证结果:")
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(
                    self.model, self.X_train, self.y_train,
                    cv=5, scoring=metric
                )
                cv_results[metric] = scores
                print(f"{metric}: {scores.mean():.4f} (±{scores.std():.4f})")
            except:
                print(f"{metric}: 无法计算")

        # 可视化交叉验证结果
        if cv_results:
            plt.figure(figsize=(10, 6))
            cv_df = pd.DataFrame(cv_results)
            sns.boxplot(data=cv_df)
            plt.title('交叉验证性能分布', fontsize=14)
            plt.ylabel('得分')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('decision_tree_cv_results.png', dpi=300, bbox_inches='tight')
            plt.show()

    def _plot_roc_curve(self, y_true, y_pred_prob):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        auc = roc_auc_score(y_true, y_pred_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'决策树 (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
        plt.xlabel('假阳性率 (FPR)')
        plt.ylabel('真阳性率 (TPR)')
        plt.title('决策树 - ROC曲线', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('decision_tree_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self, results):
        """生成分析报告"""
        print("\n=== 生成分析报告 ===")

        report_content = []
        report_content.append("# 决策树分析报告\n\n")
        report_content.append(f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_content.append(f"**数据文件**: {self.file_path}\n\n")

        # 数据概览
        report_content.append("## 1. 数据概览\n")
        report_content.append(f"- 原始数据形状: {self.data.shape}\n")
        report_content.append(f"- 特征数量: {len(self.feature_names)}\n")
        report_content.append(f"- 样本数量: {len(self.y)}\n")
        report_content.append(f"- 目标变量分布: {dict(self.y.value_counts())}\n\n")

        # 模型参数
        report_content.append("## 2. 最佳参数\n")
        for param, value in self.best_params.items():
            report_content.append(f"- {param}: {value}\n")
        report_content.append("\n")

        # 模型性能
        report_content.append("## 3. 模型性能\n")
        report_content.append(f"- 准确率: {results['accuracy']:.4f}\n")
        report_content.append(f"- AUC: {results['auc']:.4f}\n")
        report_content.append(f"- 树深度: {self.model.get_depth()}\n")
        report_content.append(f"- 叶子节点数: {self.model.get_n_leaves()}\n\n")

        # 重要特征
        report_content.append("## 4. 重要特征 (前10个)\n")
        for _, row in self.feature_importance.head(10).iterrows():
            report_content.append(f"- {row['特征']}: {row['重要性']:.4f}\n")
        report_content.append("\n")

        # 分类报告
        report_content.append("## 5. 分类报告\n")
        report_content.append("```\n")
        report_content.append(classification_report(self.y_test, results['predictions']))
        report_content.append("```\n\n")

        # 结论
        report_content.append("## 6. 结论\n")
        if results['accuracy'] > 0.8:
            report_content.append("模型性能优秀，决策树结构清晰，可用于实际预测。\n")
        elif results['accuracy'] > 0.7:
            report_content.append("模型性能良好，决策规则明确。\n")
        else:
            report_content.append("模型性能一般，建议尝试随机森林或其他集成方法。\n")

        # 保存报告
        with open('decision_tree_report.md', 'w', encoding='utf-8') as f:
            f.writelines(report_content)

        print("✓ 分析报告已保存到: decision_tree_report.md")

    def run_analysis(self, target_column):
        """运行完整的分析流程"""
        print("=" * 80)
        print("决策树分析")
        print("=" * 80)

        # 1. 加载数据
        if not self.load_data():
            return False

        # 2. 预处理数据
        if not self.preprocess_data(target_column):
            return False

        # 3. 超参数调优
        if not self.hyperparameter_tuning():
            return False

        # 4. 训练模型
        results = self.train_model()
        if results is None:
            return False

        # 5. 特征重要性分析
        self.feature_importance_analysis()

        # 6. 决策树可视化
        self.tree_visualization()

        # 7. 交叉验证
        self.cross_validation()

        # 8. 生成报告
        self.generate_report(results)

        print("\n" + "=" * 80)
        print("✓ 决策树分析完成！")
        print("=" * 80)
        print("\n生成的文件:")
        print("- decision_tree_feature_importance.csv - 特征重要性")
        print("- decision_tree_rules.txt - 决策规则")
        print("- decision_tree_report.md - 分析报告")
        print("- decision_tree_structure.png - 决策树结构")
        print("- decision_tree_roc_curve.png - ROC曲线")
        print("- decision_tree_learning_curve.png - 学习曲线")
        print("- decision_tree_parameter_tuning.png - 参数调优结果")
        print("- decision_tree_cv_results.png - 交叉验证结果")
        print("- decision_tree_feature_importance.png - 特征重要性图")

        return True


def main():
    """主函数"""
    print("=" * 80)
    print("决策树分析工具")
    print("=" * 80)

    # 创建分析器
    analyzer = DecisionTreeAnalysis()

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