#导包
from sklearn.neighbors import KNeighborsRegressor #回归模型

#准备数据集
x_train = [[0, 0, 1], [1, 1, 0], [3, 10, 10], [4, 11, 12]]  #训练集的特征
y_train = [0.1, 0.2, 0.3, 0.4]                              #训练集的标签
x_test = [[33333333, 11, 10]]                                      #测试集的特征数据， 求y_test

#创建(回归)模型对象
estimator = KNeighborsRegressor(n_neighbors=4)

#模型训练
estimator.fit(x_train, y_train)

#模型预测
y_pred = estimator.predict(x_test)

#打印预测结果
print("y_pred: ", y_pred)