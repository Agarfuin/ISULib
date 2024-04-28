from Regression.LinReg import LinReg
import pandas as pd
import numpy as np
from ANN.Model import Model
from util.util import util
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras

#####   CAR DATA MULTIPLE REGRESSION    #####

# df = pd.read_csv("Datasets/Car_Data.csv")

# X = df[['Volume', 'Weight']]
# y = df['CO2']

# regr = linear_model.LinearRegression()
# result = regr.fit(X, y)

# split_data = np.split(df, [int(.7 * len(df)), int(.85 * len(df))])
# (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = [[d[['Volume', 'Weight']].to_numpy(), d[['CO2']].to_numpy()] for d in split_data]

# epoch=1000
# result = LinReg.multiple_regression(train_x, train_y, valid_x, valid_y, epoch=epoch, learning_rate=5e-8)

# pred = result.predict(X[['Volume', 'Weight']])

# # Plot 3D graph
# fig1 = plt.figure(1)
# ax1 = fig1.add_subplot(111, projection='3d')
# ax1.plot_trisurf(X['Volume'], X['Weight'], y)
# ax1.scatter(X['Volume'], X['Weight'], pred, marker='x', c='r')
# ax1.set_xlabel('Volume')
# ax1.set_ylabel('Weight')
# ax1.set_zlabel('CO2')

# # Plot 2D graph
# fig2 = plt.figure(2)
# ax2 = fig2.add_subplot(111)
# ax2.plot(range(epoch), result.losses)
# ax2.set_xlabel('epoch')
# ax2.set_ylabel('loss')

# plt.show()




#####   CLEAN WEATHER NEURAL NETWORK    #####

data = pd.read_csv("./Datasets/clean_weather.csv", index_col=0)
data = data.ffill()

PREDICTORS = ["tmax", "tmin", "rain"]
TARGET = "tmax_tomorrow"

scaler = StandardScaler()
data[PREDICTORS] = scaler.fit_transform(data[PREDICTORS])

split_data = np.split(data, [int(.7 * len(data)), int(.85 * len(data))])
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = [[d[PREDICTORS].to_numpy(), d[[TARGET]].to_numpy()] for d in
                                                            split_data]

model = Model()
model.add_layer(3)
model.add_layer(10, 'relu')
model.add_layer(10, 'relu')
model.add_layer(1, 'relu')

model.summary()

model.fit(train_x, train_y, 5, 4, 1e-6)




#####   MNIST NEURAL NETWORK    #####

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# temp = []
# for i in range(len(y_train)):
#     temp.append(keras.utils.to_categorical(y_train[i], num_classes=10))
# y_train = np.array(temp)
# # Convert y_test into one-hot format
# temp = []
# for i in range(len(y_test)):    
#     temp.append(keras.utils.to_categorical(y_test[i], num_classes=10))
# y_test = np.array(temp)

# x_train = x_train.reshape(60000, 784)

# model = Model()
# model.add_layer(784)
# model.add_layer(5, 'relu')
# model.add_layer(10, 'softmax')
# model.summary()
# model.fit(x_train, y_train, 1, 5, 1e-6)