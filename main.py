import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from BaggedTreeRegressor import BaggedTreeRegressor

dataset_size = 100
x_axis = np.linspace(start=0, stop=2 * np.pi, num=dataset_size)
y_axis = np.sin(x_axis)

number_of_training_samples = 30
idx = np.random.choice(dataset_size, size=number_of_training_samples, replace=False)
X_train = x_axis[idx].reshape(-1, 1)
Y_train = y_axis[idx]

# NO BAGGING
model = DecisionTreeRegressor()
model.fit(X_train, Y_train)
predictionWithoutBagging = model.predict(x_axis.reshape(-1, 1))
scoreWithoutBagging = model.score(x_axis.reshape(-1, 1), y_axis)

# WITH BAGGING
modelWithBagging = BaggedTreeRegressor(number_of_sampling=200)
modelWithBagging.fit(X_train, Y_train)
predictionWithBagging = modelWithBagging.predict(x_axis.reshape(-1, 1))
scoreWithBagging = modelWithBagging.score(x_axis.reshape(-1, 1), y_axis)

# Results
print("Score without bagging : ", scoreWithoutBagging)
print("Score with bagging : ", scoreWithBagging)

plt.title("No bagging vs with bagging")
plt.plot(x_axis, y_axis, "y")
plt.plot(x_axis, predictionWithoutBagging,"r", label="No bagging")
plt.plot(x_axis, predictionWithBagging, "b", label="With bagging")
plt.legend()
plt.show()
