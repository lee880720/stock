import requests
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import time 
import datetime 
import csv

#兩日期相減 

url = "https://query1.finance.yahoo.com/v7/finance/download/%d.TW?period1=%d&period2=%d&interval=1d&events=history&includeAdjustedClose=true"

start = datetime.datetime(2020, 5, 29) 
inputstart=datetime.datetime(2020, 1, 2)  #input
startcount=((inputstart - start).days)
# print(startcount)
end = datetime.datetime(2021, 5, 28) 
inputend= datetime.datetime(2020, 1, 22)  #input
endcount=((inputend - end).days)
# print(endcount)
stocknum = 9921


for i in range(100):
    print(i)
    response = requests.get(url % (stocknum,1590710400+startcount*86400+i*86400,1622246400+endcount*86400+i*86400))
    response.text
    df = pd.read_csv(StringIO(response.text))
    # print(df)
    day=[]
    for k in range(len(df.Date)):
        day.append(k)
    print(len(df.Date))
    a=len(df.Date)
    print(df.Date[0])
    print(df.Date[a-1])

    print(len(df))
    print(type(df))
    # print(i)
    df.to_csv('ans/'+str(i)+'ans.csv')  
# plt.plot(day,df.Close)
# plt.show()