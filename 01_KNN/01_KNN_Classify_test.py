#导包

from sklearn.neighbors import KNeighborsClassifier

#准备数据集
x_train = [[0], [1], [2], [3]]
y_train = [0, 0, 1, 1]
x_test = [[5]]

#创建模型对象
estimator = KNeighborsClassifier(n_neighbors=3)

#模型训练
estimator.fit(x_train, y_train)

#模型预测
y_pre = estimator.predict(x_test)

#打印预测结果
print(f'预测值为: {y_pre}')