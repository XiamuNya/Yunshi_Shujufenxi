import pandas as pd
from sklearn.cluster import KMeans

def apply_kmeans(df, k=5):
    """
    使用K-Means算法对数据进行聚类，并将聚类标签添加到原始DataFrame中。

    Parameters:
    - df (pd.DataFrame): 包含数据的DataFrame
    - k (int): K-Means算法的聚类数量，默认为5

    Returns:
    - df (pd.DataFrame): 包含聚类标签的DataFrame
    """
    # 假设 'year' 和 'mass (g)' 是K-Means算法的相关特征
    features = df[['year', 'mass (g)']]

    # 创建K-Means模型
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(features)

    # 将聚类标签添加到原始DataFrame
    df['cluster_label'] = kmeans.labels_

    return df
