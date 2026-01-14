# -*- coding: utf-8 -*-
"""
独立的随机森林分析
包含完整的数据预处理、模型训练和结果输出
可以直接运行，不依赖其他文件
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class RandomForestAnalysis:
    """独立的随机森林分析类"""

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
        self.oob_score = None

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
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        # 创建随机森林分类器
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        # 网格搜索
        print("正在进行网格搜索...")
        grid_search = GridSearchCV(
            rf, param_grid, cv=5,
            scoring='accuracy', n_jobs=-1,
            verbose=0
        )

        try:
            grid_search.fit(self.X_train, self.y_train)
        except Exception as e:
            print(f"网格搜索失败: {e}")
            # 使用默认参数
            self.model = RandomForestClassifier(random_state=42, n_jobs=-1)
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

        # 绘制不同n_estimators的性能
        plt.subplot(1, 3, 1)
        n_estimators_results = cv_results[cv_results['param_n_estimators'].notna()]
        if len(n_estimators_results) > 0:
            n_estimators_performance = n_estimators_results.groupby('param_n_estimators')['mean_test_score'].mean()
            plt.plot(n_estimators_performance.index, n_estimators_performance.values, marker='o', markersize=8)
            plt.xlabel('树的数量')
            plt.ylabel('平均准确率')
            plt.title('不同树数量对性能的影响')
            plt.grid(True, alpha=0.3)

        # 绘制不同max_depth的性能
        plt.subplot(1, 3, 2)
        depth_results = cv_results[cv_results['param_max_depth'].notna()]
        if len(depth_results) > 0:
            depth_performance = depth_results.groupby('param_max_depth')['mean_test_score'].mean()
            plt.plot(depth_performance.index, depth_performance.values, marker='s', markersize=8)
            plt.xlabel('最大深度')
            plt.ylabel('平均准确率')
            plt.title('不同树深度对性能的影响')
            plt.grid(True, alpha=0.3)

        # 绘制不同max_features的性能
        plt.subplot(1, 3, 3)
        features_results = cv_results[cv_results['param_max_features'].notna()]
        if len(features_results) > 0:
            features_performance = features_results.groupby('param_max_features')['mean_test_score'].mean()
            colors = ['skyblue', 'lightcoral']
            bars = plt.bar(range(len(features_performance)), features_performance.values, color=colors[:len(features_performance)])
            plt.xticks(range(len(features_performance)), features_performance.index)
            plt.xlabel('最大特征数')
            plt.ylabel('平均准确率')
            plt.title('不同最大特征数对性能的影响')
            plt.grid(True, alpha=0.3)
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('random_forest_parameter_tuning.png', dpi=300, bbox_inches='tight')
        plt.show()

    def train_model(self):
        """训练最终模型"""
        print("\n=== 训练最终模型 ===")

        # 使用最佳参数训练模型（启用OOB评分）
        final_params = self.best_params.copy()
        final_params['bootstrap'] = True  # 确保启用bootstrap以计算OOB
        final_params['oob_score'] = True

        self.model = RandomForestClassifier(random_state=42, n_jobs=-1, **final_params)

        self.model.fit(self.X_train, self.y_train)

        # 获取OOB分数
        if hasattr(self.model, 'oob_score_'):
            self.oob_score = self.model.oob_score_
            print(f"OOB得分: {self.oob_score:.4f}")

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
        plt.title('随机森林学习曲线', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('random_forest_learning_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

    def feature_importance_analysis(self):
        """特征重要性分析"""
        print("\n=== 特征重要性分析 ===")

        # 1. 基于不纯度的特征重要性
        importance_impurity = pd.DataFrame({
            '特征': self.feature_names,
            '不纯度重要性': self.model.feature_importances_
        }).sort_values('不纯度重要性', ascending=False)

        print("\n基于不纯度的特征重要性:")
        print(importance_impurity.round(4))

        # 2. 基于排列的特征重要性
        print("\n计算排列重要性...")
        try:
            perm_importance = permutation_importance(
                self.model, self.X_test, self.y_test,
                n_repeats=10, random_state=42, n_jobs=-1
            )

            importance_permutation = pd.DataFrame({
                '特征': self.feature_names,
                '排列重要性': perm_importance.importances_mean
            }).sort_values('排列重要性', ascending=False)

            print("\n基于排列的特征重要性:")
            print(importance_permutation.round(4))

            # 合并重要性结果
            combined_importance = importance_impurity.merge(
                importance_permutation, on='特征', how='outer'
            ).fillna(0)

        except Exception as e:
            print(f"排列重要性计算失败: {e}")
            combined_importance = importance_impurity
            combined_importance['排列重要性'] = 0

        self.feature_importance = combined_importance

        # 可视化特征重要性
        self._plot_feature_importance(importance_impurity, importance_permutation if 'importance_permutation' in locals() else None)

        # 保存特征重要性
        combined_importance.to_csv('random_forest_feature_importance.csv', index=False, encoding='utf-8-sig')
        print("\n特征重要性已保存到: random_forest_feature_importance.csv")

    def _plot_feature_importance(self, importance_impurity, importance_permutation=None):
        """可视化特征重要性"""
        plt.figure(figsize=(15, 10))

        # 不纯度重要性
        plt.subplot(2, 2, 1)
        top_features = importance_impurity.head(15)
        bars = sns.barplot(data=top_features, x='不纯度重要性', y='特征')
        plt.title('基于不纯度的特征重要性 (前15个)', fontsize=14)
        plt.xlabel('重要性')
        # 添加数值标签
        for i, bar in enumerate(bars.patches):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        plt.tight_layout()

        # 排列重要性
        if importance_permutation is not None:
            plt.subplot(2, 2, 2)
            top_features_perm = importance_permutation.head(15)
            bars = sns.barplot(data=top_features_perm, x='排列重要性', y='特征')
            plt.title('基于排列的特征重要性 (前15个)', fontsize=14)
            plt.xlabel('重要性')
            # 添加数值标签
            for i, bar in enumerate(bars.patches):
                width = bar.get_width()
                plt.text(width + 0.0001, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
            plt.tight_layout()

        # 累积重要性
        plt.subplot(2, 2, 3)
        cumulative_importance = np.cumsum(importance_impurity['不纯度重要性'])
        plt.plot(range(len(cumulative_importance)), cumulative_importance, marker='o', markersize=4)
        plt.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90%累积重要性')
        plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%累积重要性')
        plt.xlabel('特征数量')
        plt.ylabel('累积重要性')
        plt.title('特征累积重要性', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 重要性对比（如果有两种重要性）
        if importance_permutation is not None:
            plt.subplot(2, 2, 4)
            common_features = importance_impurity.head(10)['特征'].tolist()
            comparison_df = self.feature_importance[self.feature_importance['特征'].isin(common_features)]

            x = np.arange(len(comparison_df))
            width = 0.35

            plt.bar(x - width/2, comparison_df['不纯度重要性'], width, label='不纯度重要性', alpha=0.8)
            plt.bar(x + width/2, comparison_df['排列重要性'], width, label='排列重要性', alpha=0.8)
            plt.xlabel('特征')
            plt.ylabel('重要性')
            plt.title('不同重要性度量对比', fontsize=14)
            plt.xticks(x, comparison_df['特征'], rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def tree_analysis(self):
        """分析森林中的树"""
        print("\n=== 森林中的树分析 ===")

        # 分析树的深度分布
        tree_depths = [tree.get_depth() for tree in self.model.estimators_]
        tree_nodes = [tree.tree_.node_count for tree in self.model.estimators_]

        print(f"树的数量: {len(self.model.estimators_)}")
        print(f"平均树深度: {np.mean(tree_depths):.2f} (±{np.std(tree_depths):.2f})")
        print(f"平均节点数: {np.mean(tree_nodes):.2f} (±{np.std(tree_nodes):.2f})")

        # 可视化树的分布
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.hist(tree_depths, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        plt.xlabel('树深度')
        plt.ylabel('频数')
        plt.title('森林中树的深度分布')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.hist(tree_nodes, bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
        plt.xlabel('节点数')
        plt.ylabel('频数')
        plt.title('森林中树的节点数分布')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('random_forest_trees_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 可视化几棵代表性的树
        n_trees_to_show = min(3, len(self.model.estimators_))
        for i in range(n_trees_to_show):
            plt.figure(figsize=(20, 10))
            plot_tree(
                self.model.estimators_[i],
                feature_names=self.feature_names,
                class_names=['0', '1'],  # 假设二分类
                filled=True,
                rounded=True,
                max_depth=3,
                fontsize=10,
                impurity=False
            )
            plt.title(f'随机森林中的第{i+1}棵树 (前3层)', fontsize=18)
            plt.savefig(f'random_forest_tree_{i+1}.png', dpi=300, bbox_inches='tight')
            plt.show()

    def ensemble_analysis(self):
        """集成分析"""
        print("\n=== 集成分析 ===")

        # 获取每棵树的预测
        tree_predictions = []
        for tree in self.model.estimators_:
            tree_pred = tree.predict(self.X_test)
            tree_predictions.append(tree_pred)

        tree_predictions = np.array(tree_predictions)

        # 分析预测一致性
        prediction_consistency = []
        for i in range(len(self.X_test)):
            unique_preds, counts = np.unique(tree_predictions[:, i], return_counts=True)
            majority_vote = unique_preds[np.argmax(counts)]
            consistency = np.max(counts) / len(tree_predictions)
            prediction_consistency.append(consistency)

        avg_consistency = np.mean(prediction_consistency)
        print(f"平均预测一致性: {avg_consistency:.4f}")

        # 分析不同样本的预测难度
        difficult_samples = np.where(np.array(prediction_consistency) < 0.7)[0]
        print(f"预测困难的样本数 (一致性<0.7): {len(difficult_samples)}")

        # 可视化预测一致性
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.hist(prediction_consistency, bins=20, alpha=0.7, edgecolor='black', color='gold')
        plt.xlabel('预测一致性')
        plt.ylabel('频数')
        plt.title('预测一致性分布')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.scatter(range(len(prediction_consistency)), prediction_consistency, alpha=0.5, s=10)
        plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='困难样本阈值')
        plt.xlabel('样本索引')
        plt.ylabel('预测一致性')
        plt.title('样本预测一致性')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('random_forest_ensemble_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

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
            plt.savefig('random_forest_cv_results.png', dpi=300, bbox_inches='tight')
            plt.show()

    def _plot_roc_curve(self, y_true, y_pred_prob):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        auc = roc_auc_score(y_true, y_pred_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'随机森林 (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
        plt.xlabel('假阳性率 (FPR)')
        plt.ylabel('真阳性率 (TPR)')
        plt.title('随机森林 - ROC曲线', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('random_forest_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self, results):
        """生成分析报告"""
        print("\n=== 生成分析报告 ===")

        report_content = []
        report_content.append("# 随机森林分析报告\n\n")
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
        if self.oob_score is not None:
            report_content.append(f"- OOB得分: {self.oob_score:.4f}\n")
        report_content.append(f"- 树的数量: {self.model.n_estimators}\n")
        report_content.append(f"- 平均树深度: {np.mean([tree.get_depth() for tree in self.model.estimators_]):.2f}\n\n")

        # 重要特征
        report_content.append("## 4. 重要特征 (前10个)\n")
        for _, row in self.feature_importance.head(10).iterrows():
            report_content.append(f"- {row['特征']}: {row['不纯度重要性']:.4f}\n")
        report_content.append("\n")

        # 分类报告
        report_content.append("## 5. 分类报告\n")
        report_content.append("```\n")
        report_content.append(classification_report(self.y_test, results['predictions']))
        report_content.append("```\n\n")

        # 结论
        report_content.append("## 6. 结论\n")
        if results['accuracy'] > 0.8:
            report_content.append("模型性能优秀，集成学习效果显著，可用于实际预测。\n")
        elif results['accuracy'] > 0.7:
            report_content.append("模型性能良好，随机森林提供了稳健的预测能力。\n")
        else:
            report_content.append("模型性能一般，建议调整参数或进行特征工程。\n")

        # 保存报告
        with open('random_forest_report.md', 'w', encoding='utf-8') as f:
            f.writelines(report_content)

        print("✓ 分析报告已保存到: random_forest_report.md")

    def run_analysis(self, target_column):
        """运行完整的分析流程"""
        print("=" * 80)
        print("随机森林分析")
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

        # 6. 森林中的树分析
        self.tree_analysis()

        # 7. 集成分析
        self.ensemble_analysis()

        # 8. 交叉验证
        self.cross_validation()

        # 9. 生成报告
        self.generate_report(results)

        print("\n" + "=" * 80)
        print("✓ 随机森林分析完成！")
        print("=" * 80)
        print("\n生成的文件:")
        print("- random_forest_feature_importance.csv - 特征重要性")
        print("- random_forest_report.md - 分析报告")
        print("- random_forest_roc_curve.png - ROC曲线")
        print("- random_forest_learning_curve.png - 学习曲线")
        print("- random_forest_parameter_tuning.png - 参数调优结果")
        print("- random_forest_trees_distribution.png - 树的分布")
        print("- random_forest_tree_*.png - 森林中的树")
        print("- random_forest_ensemble_analysis.png - 集成分析")
        print("- random_forest_cv_results.png - 交叉验证结果")
        print("- random_forest_feature_importance.png - 特征重要性图")

        return True


def main():
    """主函数"""
    print("=" * 80)
    print("随机森林分析工具")
    print("=" * 80)

    # 创建分析器
    analyzer = RandomForestAnalysis()

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