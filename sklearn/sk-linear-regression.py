import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# # Load diabetes dataset
# diabetes = datasets.load_diabetes()

# # Use only one feature
# data_x = diabetes.data[:, np.newaxis, 2]

# # Split the data into training/testing sets
# data_x_train = data_x[:-20]
# data_x_test = data_x[-20:]

# # Split targets into training/testing sets
# data_y_train = diabetes.target[:-20]
# data_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
data = np.loadtxt('../data/aerogerador.dat')
data_x_train = data[:,[0]]
data_y_train = data[:,[1]]
print(data_x_train.shape)
print(data_y_train.shape)
regr.fit(data_x_train, data_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f' % np.mean((regr.predict(data_x_train) - data_y_train) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(data_x_train, data_y_train))

# Plot outputs
plt.scatter(data_x_train, data_y_train, color='black')
plt.plot(data_x_train, regr.predict(data_x_train), color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()