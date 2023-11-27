'''主函数'''
import warnings
from data_processing import load_and_explore_data, clean_data, standardize_data
from visualization import (
    visualize_mass_vs_year, visualize_correlation_heatmap, visualize_kmeans_clusters
)
from modeling import train_and_evaluate_model
from matplotlib import pyplot as plt

# 忽略FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决符号显示问题

if __name__ == "__main__":
    file_path = 'templates/data/Meteorite_Landings.csv'

    # 加载和探索数据
    meteorite_df = load_and_explore_data(file_path)

    # 数据清洗
    meteorite_df = clean_data(meteorite_df)

    # 数据标准化
    meteorite_df = standardize_data(meteorite_df)

    # 可视化陨石质量与年份关系
    visualize_mass_vs_year(meteorite_df, save_path='templates/plt_imgs/visualization.png')

    # 聚类结果可视化
    visualize_kmeans_clusters(meteorite_df, save_path='templates/plt_imgs/visualize_kmeans_clusters.png')

    # 可视化相关性矩阵
    visualize_correlation_heatmap(meteorite_df, save_path='templates/plt_imgs/visualize_correlation_heatmap.png')

    # 训练并评估模型
    train_and_evaluate_model(meteorite_df, save_path='templates/plt_imgs/train_and_evaluate_model.png')
