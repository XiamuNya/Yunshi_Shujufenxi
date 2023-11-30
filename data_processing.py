import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''数据处理文件'''

def load_and_explore_data(file_path):
    """
    读取CSV文件，探索数据并返回DataFrame。

    Parameters:
    - file_path (str): 文件路径

    Returns:
    - df (pd.DataFrame): 包含数据的DataFrame
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 打印前10行数据
    print(df.head(10))
    # 打印数据信息
    print(df.info())
    # 打印数据形状
    print('Shape:', df.shape)
    # 打印数据描述统计信息
    print(df.describe())
    return df

def clean_data(df):
    """
    清理数据，删除包含缺失值的行。

    Parameters:
    - df (pd.DataFrame): 包含数据的DataFrame

    Returns:
    - df (pd.DataFrame): 清理后的DataFrame
    """
    # 删除包含缺失值的行（'mass (g)', 'year', 'reclat', 'reclong' 列）
    df.dropna(subset=['mass (g)', 'year', 'reclat', 'reclong'], inplace=True)
    return df

def standardize_data(df):
    """
    标准化数据，使用StandardScaler对'mass (g)'列进行标准化。

    Parameters:
    - df (pd.DataFrame): 包含数据的DataFrame

    Returns:
    - df (pd.DataFrame): 标准化后的DataFrame
    """
    # 初始化StandardScaler
    scaler = StandardScaler()
    # 对'mass (g)'列进行标准化
    df[['mass (g)']] = scaler.fit_transform(df[['mass (g)']])
    return df

def standardize_and_train_model(df):
    """
    标准化数据，创建线性回归模型并进行训练。

    Parameters:
    - df (pd.DataFrame): 包含数据的DataFrame

    Returns:
    - model: 训练好的线性回归模型
    """
    # 标准化数据
    df = standardize_data(df)

    # 创建线性回归模型
    model = LinearRegression()

    # 特征选择
    X = df[['year']]
    y = df['mass (g)']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    return model, X_test, y_test