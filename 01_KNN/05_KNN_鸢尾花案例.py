#通过KNN算法实现 鸢尾花分类
# 机器学习项目研发的一般流程
# 1，加载数据
# 2，数据预处理
# 3，特征工程
#     特征提取
#     预处理
#     。。。
# 4，模型训练
# 5，模型评估
# 6，模型预测

# 导包
from sklearn.datasets import load_iris  #加载鸢尾花测试集
import seaborn as sns                   #
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    #分割训练集和测试集
from sklearn.preprocessing import StandardScaler        #数据标准化
from sklearn.neighbors import KNeighborsClassifier      #KNN算法分类对象
from sklearn.metrics import accuracy_score              #模型评估

#1，定义函数，加载鸢尾花
def fun_load_iris():
    iris_data = load_iris()
    #print(f'数据集:{iris_data}') #字典类型
    #print(f'数据集的类型:{type(iris_data)}')



    # 数据集中所有的键
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
    #data 数据
    #feature_names 特征名
    #target 标签的key 0,1,2
    #target_names 标签的value ['setosa' 'versicolor' 'virginica']

    # print('数据集中所有的键',iris_data.keys())
    # print(f'具体的数据：\n{iris_data.data[0:5]}') #取前5条数据，总共150条数据
    # print(f'特征的名字：{iris_data.feature_names}')
    # print(f'花的类型名key：{iris_data.target[:5]}')
    # print(f'花的类型名value：{iris_data.target_names}')
    # print(f'数据的描述信息:\n{iris_data.DESCR}')
    # print(f'数据集的框架:{iris_data.frame}')

#2，定义函数，绘制数据集的散点图
def fun_show_iris():
    #1，加载数据集
    iris_data = load_iris()

    #2，把数据集封装为DataFrame
    iris_df = pd.DataFrame(data = iris_data.data, columns = iris_data.feature_names)

    #3,新增一列->标签列
    iris_df['label'] = iris_data.target

    #4,通过Seanborn绘制散点图
    sns.lmplot(iris_df, x = 'sepal length (cm)', y = 'sepal width (cm)', hue = 'label', fit_reg = True )  #根据label分组

    #5,设置标题显示
    plt.title('iris data') #自动调整子图参数，以使整个给图像的边界与子图匹配
    plt.tight_layout()
    plt.show()
    print(iris_df)

#3，定义函数，切分训练集和数据集
def fun_split_train_test():
    #加载数据集
    iris_data = load_iris()

    #数据的预处理，从150个特征和标签中，按照8：2的比例切分训练集和测试集
                    # 元组（Tuple）是Python中一种非常重要的数据结构。让我用简单的语言解释：
                    #
                    # 什么是元组？
                    # 元组是一个不可变的、有序的元素集合。
                    #
                    # 基本特点：
                    # 📝 不可变：创建后不能修改（不能增、删、改元素）
                    #
                    # 🔢 有序：元素有固定的顺序
                    #
                    # 📦 可以存储任意类型：数字、字符串、列表等都可以
                    # # 方法1：使用圆括号
                    # tuple1 = (1, 2, 3, 4)
                    # tuple2 = ("苹果", "香蕉", "橙子")
                    #
                    # # 方法2：不使用括号（逗号分隔）
                    # tuple3 = 1, 2, 3  # 自动变成元组 (1, 2, 3)
                    #
                    # # 方法3：单个元素的元组（必须加逗号）
                    # single_tuple = (5,)  # 这是元组
                    # not_tuple = (5)  # 这只是数字5，不是元组！
                    #
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size = 0.2, random_state = 23)
    print(f'训练集的特征:{x_train}, 个数：{len(x_train)}')
    print(f'训练集的标签:{y_train}, 个数：{len(y_train)}')
    print(f'测试集的特征:{x_test}, 个数：{len(x_test)}')
    print(f'测试集的标签:{y_test}, 个数：{len(y_test)}')

#4，定义函数，实现鸢尾花完整案例-> 加载数据，数据预处理，特征工程，模型训练，模型评估，模型预测
def fun_iris_evaluate_test():
    #1, 加载数据集
    iris_data = load_iris()

    #2，数据预处理
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size = 0.2, random_state = 22)

    #3,特征工程(提取，预处理)
        #特征提取：因为源数据只有四个特征列，且都是我们要用的，所以无需特征提取
        #特征预处理:因为源数据的四列特征的插值不大，所以我们无需预处理，但是为了代码完整，我们进行预处理
        #标准化
    scaler = StandardScaler()#标准化对象
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #4,模型训练
    estimator = KNeighborsClassifier(n_neighbors = 3)
    estimator.fit(x_train, y_train)

    #5,模型预测(切分的测试集)
    y_pred = estimator.predict(x_test)
    print(f"切分预测值为:{y_pred}")

    #5,新数据的预测
    my_data = [[7.8, 2.1, 3.9, 1.6]]
    my_data = scaler.transform(my_data)
    y_pred_new = estimator.predict(my_data)
    print(f"y_pred_new:{y_pred_new}")
    print(f"y_pred_new_score:{estimator.predict_proba(my_data)}")

    #6，模型评估
        #方式1：直接评分，基于训练集的特征和标签
    print(f'正确率(准确率)：{estimator.score(x_train, y_train)}')

        #方式2：
    print(f'准确率(正确率){accuracy_score(y_test, estimator.predict(x_test))}')





if __name__ == '__main__':
    # fun_load_iris()
    # fun_show_iris()
    fun_iris_evaluate_test()