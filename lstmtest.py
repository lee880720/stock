from keras.models import load_model
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import pandas
import numpy

import requests
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import time 
import datetime 
import csv
#######################################################無意義但不能刪
dataset = pandas.read_csv('9921.csv')
dataset = dataset.Close.values.astype('float32')
# dataset = dataset.reshape(241, 1)
dataset = dataset.astype('float32')
dataset = dataset.reshape(len(dataset), 1)
# 正規化(normalize) 資料，使資料值介於[0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
######################################################無意義但不能刪
url = "https://query1.finance.yahoo.com/v7/finance/download/%d.TW?period1=%d&period2=%d&interval=1d&events=history&includeAdjustedClose=true"

predday = datetime.datetime(2021, 6,3)    #輸入預測日期
stocknum = 9921    #股票代碼 

start = datetime.datetime(2020, 5, 29)  # 1590710400

count = ((predday-start).days)    #日期差

tenday = 10   #拿去減 但是會有錯 因為要扣掉休市(假日 連假 及 過年可能沒有開市)

response = requests.get(url % (stocknum,1590710400+(count-tenday)*86400,1590710400+count*86400))

response.text

df = pd.read_csv(StringIO(response.text))

while len(df)!=11:      #算出有開市的10天 
    tenday = tenday+1
    response = requests.get(url % (stocknum,1590710400+(count-tenday)*86400,1590710400+count*86400))
    response.text
    df = pd.read_csv(StringIO(response.text))



model = load_model('a2016-2021_9921.h5')     #載入之前存的model

dataset_total = df   #讀入資料10天資料
print(df)

dataset_total = dataset_total.Close.values.astype('float32')  #轉成小數

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

print()
print("我預測"+str(predday)+"後一天價格為"+str(predicted_stock_price))
