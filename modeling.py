'''建模文件'''
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error



def train_and_evaluate_model(df, save_path=None):
    """
       使用线性回归模型对陨石质量进行预测，并可视化预测结果。

       Parameters:
       - df (pd.DataFrame): 包含数据的DataFrame
       """
    X = df[['year']]
    y = df['mass (g)']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 显式设置 n_init 参数，避免警告
    kmeans = KMeans(n_clusters=5, n_init=10)
    df['cluster_label'] = kmeans.fit_predict(df[['year', 'mass (g)']])

    # 构建线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 模型预测
    y_pred = model.predict(X_test)

    # 计算均方误差
    mse = mean_squared_error(y_test, y_pred)
    print(f'均方误差: {mse}')

    # 可视化线性回归模型预测结果
    plt.figure(figsize=(12, 8))
    plt.scatter(X_test, y_test, color='blue', label='实际值')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='预测值')
    plt.xlabel('年份')
    plt.ylabel('陨石质量 (g)')
    plt.legend()
    plt.title('线性回归模型预测陨石质量')
    # 保存图表为图像文件（如果提供了保存路径）
    if save_path:
        plt.savefig(save_path)
    plt.show()



def evaluate_model_performance(model, X_test, y_test):
    """
    评估模型性能，打印 R^2 分数和平均绝对误差。

    Parameters:
    - model: 训练好的模型
    - X_test: 测试集特征
    - y_test: 测试集目标值
    """
    # 模型预测
    y_pred = model.predict(X_test)

    # 计算 R^2 分数
    r2 = r2_score(y_test, y_pred)
    print(f'R^2 分数: {r2}')

    # 计算平均绝对误差
    mae = mean_absolute_error(y_test, y_pred)
    print(f'平均绝对误差: {mae}')