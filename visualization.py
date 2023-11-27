# visualization.py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def visualize_mass_vs_year(df, save_path=None):
    """
       绘制陨石质量与年份的散点图。

       Parameters:
       - df (pd.DataFrame): 包含数据的DataFrame
       """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='year', y='mass (g)', data=df)
    plt.title('陨石质量与年份关系')

    # 保存图表为图像文件（如果提供了保存路径）
    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_correlation_heatmap(df, save_path=None):
    """
        计算相关性矩阵并绘制热力图。

        Parameters:
        - df (pd.DataFrame): 包含数据的DataFrame
        """
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('相关矩阵热力图')
    # 保存图表为图像文件（如果提供了保存路径）
    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_kmeans_clusters(df, save_path=None):
    """
        假设K-Means聚类的结果已经保存在'cluster_label'列中，绘制K-Means聚类结果的散点图。

        如果'cluster_label'列不存在，执行K-Means聚类。

        Parameters:
        - df (pd.DataFrame): 包含数据的DataFrame
        """
    # 假设 K-Means 聚类的结果已经保存在 'cluster_label' 列中
    if 'cluster_label' not in df.columns:
        # 如果 'cluster_label' 列不存在，执行 K-Means 聚类
        kmeans = KMeans(n_clusters=5)  # 设置聚类的个数
        df['cluster_label'] = kmeans.fit_predict(df[['year', 'mass (g)']])

    # 绘制散点图
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='year', y='mass (g)', hue='cluster_label', data=df, palette='viridis', legend='full')
    plt.title('K-Means 聚类结果可视化')
    # 保存图表为图像文件（如果提供了保存路径）
    if save_path:
        plt.savefig(save_path)
    plt.show()