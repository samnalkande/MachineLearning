import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
# sam is a variable
sam_x = np.array([[1],[2],[3]])

sam_x_train = sam_x
sam_x_test = sam_x

sam_y_train = np.array([3,2,4])
sam_y_test = np.array([3,2,4])

model = linear_model.LinearRegression()

model.fit(sam_x_train, sam_y_train)

sam_y_predict = model.predict(sam_x_test)

print("Mean squared error is : ", mean_squared_error(sam_y_test, sam_y_predict))

print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

plt.scatter(sam_x_test, sam_y_test)
plt.plot(sam_x_test, sam_y_predict)

plt.show()
