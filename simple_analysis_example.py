# -*- coding: utf-8 -*-
"""
简化版数据分析示例 - 用于快速测试
假设数据中有一个名为 'target' 的目标变量列
"""

from data_analysis import DataAnalyzer

def quick_analysis():
    """快速分析示例"""
    # 数据文件路径
    file_path = r"C:\Users\霍冠华\Desktop\毕设\数据处理\处理后数据_20260104_144216.xlsx"

    # 创建分析器
    analyzer = DataAnalyzer(file_path)

    # 尝试使用常见的列名作为目标变量
    possible_targets = ['target', 'label', 'class', 'outcome', 'result', 'y', '类别', '目标', '分类']

    # 加载数据
    if analyzer.load_data():
        print("\n数据列名:")
        print(analyzer.data.columns.tolist())

        # 查找可能的目标变量
        target_found = False
        for target in possible_targets:
            if target in analyzer.data.columns:
                print(f"\n找到目标变量: {target}")
                target_found = True
                analyzer.run_complete_analysis(target)
                break

        if not target_found:
            print("\n未找到常见的目标变量列名，请手动指定")
            target = input("请输入目标变量列名: ")
            if target and target in analyzer.data.columns:
                analyzer.run_complete_analysis(target)
            else:
                print("指定的列名不存在，分析终止")

if __name__ == "__main__":
    quick_analysis()