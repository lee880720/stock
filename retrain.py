import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import pandas
import matplotlib.pyplot as plt
# 載入訓練資料
dataset = pandas.read_csv('2016-2021_9921.csv')
dataset = dataset.Close.values.astype('float32')
# dataset = dataset.reshape(241, 1)
dataset = dataset.astype('float32')
dataset = dataset.reshape(len(dataset), 1)
# 正規化(normalize) 資料，使資料值介於[0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

X_train = [] 
y_train = []
train_set=dataset
training_set_scaled=dataset
for i in range(10,len(train_set)):
    X_train.append(training_set_scaled[i-10:i-1, 0]) 
    y_train.append(training_set_scaled[i, 0]) 
X_train, y_train = numpy.array(X_train), numpy.array(y_train) 
X_train = numpy.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print(len(X_train))
print(len(y_train))

print(X_train[1])
print(y_train[1])

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout,BatchNormalization

# keras.backend.clear_session()
# regressor = Sequential()
# regressor.add(LSTM(units = 100, input_shape = (X_train.shape[1], 1)))
# regressor.add(Dense(units = 1))
# regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
model = load_model('2016-2021_9921.h5')

history = model.fit(X_train, y_train, epochs = 25, batch_size = 8)

model.save('a2016-2021_9921.h5')


dataset_total = pandas.read_csv('9921.csv')
dataset_total = dataset_total.Close.values.astype('float32')

inputs = dataset_total

inputs = inputs.reshape(-1,1)

inputs = scaler.transform(inputs)

X_test = []
for i in range(10, len(inputs)):
    X_test.append(inputs[i-10:i-1, 0]) 
X_test = numpy.array(X_test)

X_test = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = model.predict(X_test)

predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

print(predicted_stock_price)


a=predicted_stock_price[0]

for i in range(10):
    predicted_stock_price=numpy.insert(predicted_stock_price,0,[a],0)


#使用sc的 inverse_transform將股價轉為歸一化前

# print(predicted_stock_price)
print(len(X_test))
print(len(predicted_stock_price))
plt.plot(dataset_total, color = 'black', label = 'Real TSMC Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TSMC Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()