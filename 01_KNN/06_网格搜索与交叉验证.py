from sklearn.datasets import load_iris  #加载鸢尾花测试集
from sklearn.model_selection import train_test_split, GridSearchCV    #分割训练集和测试集
from sklearn.preprocessing import StandardScaler        #数据标准化
from sklearn.neighbors import KNeighborsClassifier      #KNN算法分类对象
from sklearn.metrics import accuracy_score              #模型评估

# 1，加载数据集
iris_data = load_iris()

# 2，数据预处理（这里是切分训练集和测试集8：2)
x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size = 0.2, random_state = 22)

#3，特征工程（标准化)
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

#4，模型训练
    ##差异点
estimator = KNeighborsClassifier()
param_dict = {"n_neighbors":[i for i in range(1, 11)]}

#创建GRidSearchCV对象 寻找最优超参，使用网格搜索加交叉验证
    #参1：要计算最优超参的模型对象
    #参2：超参字典
    #参3：交叉验证的折数，每个超参都会进行4次交叉验证， 这里是 4*10 = 40次
    #返回值：是找到最优参数组合的模型对象
estimator = GridSearchCV(estimator, param_dict, cv = 20)
estimator.fit(x_train, y_train)
#打印最优超参组合

print(f"最优评分：{estimator.best_score_}")
print(f'最优超参组合：{estimator.best_params_}')
print(f'最优的估计器对象：{estimator.best_estimator_}')
print(f'具体的交叉验证结果；{estimator.cv_results_}')

#验证
#estimator = estimator.best_estimator_   #获取最优模型对象
estimator = KNeighborsClassifier(n_neighbors = 1)
estimator.fit(x_train, y_train)         #模型训练



#模型评估
print(f"模型评估(准确率)：{accuracy_score(y_test, estimator.predict(x_test))}")
