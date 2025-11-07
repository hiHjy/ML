# 28*28 的图片
# 导包
from importlib.metadata import pass_none

from sklearn.datasets import load_iris  #加载鸢尾花测试集
import seaborn as sns                   #
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    #分割训练集和测试集
from sklearn.preprocessing import StandardScaler        #数据标准化
from sklearn.neighbors import KNeighborsClassifier      #KNN算法分类对象
from sklearn.metrics import accuracy_score              #模型评估
import joblib #保存模型
from collections import Counter

#1 显示数字
def show_digit(idx):
    df = pd.read_csv("./data/手写数字识别.csv")
    #print(df)
    print(f"一共{len(df)}行")

    if idx < 0 or idx > len(df) - 1:
        print("索引越界")
        return

    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    print(f"查看所有标签:{Counter(y)}")
    print(f"该图片对应的数字应该是:{y.iloc[idx]}")

    #查看一下x的形状
    print(f"x.iloc[idx]的形状:{x.iloc[idx].shape}")
    print(f"_train.iloc[idx]的数据:{x.iloc[idx].values}")

    x = x.iloc[idx].values.reshape(28, 28)
    print( x.shape)

    #绘制数字
    plt.imshow(x, cmap = 'gray')#灰度图
    plt.axis("off")             #关闭坐标轴
    plt.show()


#2 (掌握) 模型训练，并保存训练号的模型
def train_model():
    #1 加载数据集
    df = pd.read_csv("./data/手写数字识别.csv")

    #2 数据预处理（划分数据集）
    x = df.iloc[:, 1:]  #特征
    y = df.iloc[:, 0]   #标签

    #打印特征和标签的形状
    print(f"shape:x:  {x.shape}, y:{y.shape}")

    #打印标签的分布情况
    print(f"标签的分布情况:{Counter(y)}")
    #在分割数据集前先做归一化
    x = (x - 0) / (255 -  0)



    #拆分训练集和测试集
        # x: 特征
        # y:标签
        # test_size: 训练集：测试集 = 4：1
        # random_state:随机数种子
        # stratify: 根据标签（y）的分布去划分训练集和测试集防止出现某个标签没有被作为训练集

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 22, stratify = y)

    #3 模型训练
    estimator = KNeighborsClassifier(n_neighbors=3)

    estimator.fit(x_train, y_train)


    #5 模型评估
    print(f"准确率：{estimator.score(x_test, y_test)}")
    print(f"准确率：{accuracy_score(y_test, estimator.predict(x_test))}")

    # 保存模型
    joblib.dump(estimator, "./my_model/手写数字识别.pkl") #pickle ptn 都可以
    print("模型保存成功")


#3 (掌握) 测试模型
def use_model():
    #1 加载图片
    img_x = plt.imread("./data/demo2.png") # pix 28*28
    # 如果是RGB图，取单通道
    if len(img_x.shape) == 3:
        img_x = img_x[:, :, 0]


    print("img.shape:", img_x.shape, " dtype:", img_x.dtype, " min:", img_x.min(), " max:", img_x.max())

    # 转换数据格式(并归一化）
    img_x = img_x.reshape(1, -1)  #语法糖，效果同img_x.reshape(1, 784) 转换为:1*784
    img_x = 1 - img_x
    # #2 绘制图片
    # plt.imshow(img_x, "gray")
    # plt.show()

    #3 加载模型
    estimator = joblib.load("./my_model/手写数字识别.pkl")

    #4 模型预测
    y_pred = estimator.predict(img_x)

    print(f"图片中的数字为:{y_pred}")


#main

if __name__ == '__main__':
   # show_digit(17)
   #train_model()
   use_model()