#导包
from sklearn.linear_model import LinearRegression

#1 加载数据集
x_train = [[160], [166], [172], [174], [180]]
y_train = [56.3, 60.6, 65.1, 68.5, 75]
x_test = [[176]]

#2 数据预处理

#3 特征工程(提取，预处理)

#4 训练模型
estimator = LinearRegression()
estimator.fit(x_train, y_train)

    #查看一下权重和偏置
w = estimator.coef_
b = estimator.intercept_

print(f"w:{w}, b:{b}")

#5 模型预测
y_pred = estimator.predict(x_test)
print(y_pred)

#6 模型评估