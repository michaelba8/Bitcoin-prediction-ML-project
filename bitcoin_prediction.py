import json
import requests
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA



def main():
    result=read_data()
    print(result.shape)
    """constant define"""
    openTime=0
    openPrice=1
    highPrice=2
    lowPrice=3
    closePrice=4
    volume=5
    closeTime=6
    quoteAV=7
    trades=8
    takerBaseAV=9
    takerQuoteAV=10




def read_data(months=20):
    """reading the Bitcoin to Dollar of the last #months (maximum=21)"""
    if months>20:
        months=20
    const="data/BTCTUSD-1m-"
    year=2020
    month=10
    s_month=str(10)
    count=1
    reader = csv.reader(open("data/BTCTUSD-1m-2020-10.csv", "rt"), delimiter=",")
    temp = list(reader)
    result = np.array(temp).astype("float")
    print("append file "+ "data/BTCTUSD-1m-2020-10.csv")
    while(count<months):
        month-=1
        if(month<1):
            month=12
            year-=1

        if(month<10):
            s_month="0"+str(month)
        else:
            s_month=str(month)
        s=const+str(year)+"-"+s_month+".csv"
        reader = csv.reader(open(s, "rt"), delimiter=",")
        temp = list(reader)
        res = np.array(temp).astype("float64")
        result=np.vstack((res,result))
        count+=1
        print("append file",s)

    return result[:,0:-1] ## delete the last column it is useless


def get_bars(symbol="BTCUSDT", interval="1m", limit="1500"):
    url = 'https://fapi.binance.com/fapi/v1/klines' + '?symbol=' + symbol + '&interval=' + interval + '&limit=' + limit
    data = json.loads(requests.get(url).text)
    df = pd.DataFrame(data)
    df.columns = ['open_time',
                  'o', 'h', 'l', 'c', 'v',
                  'close_time', 'qav', 'num_trades',
                  'taker_base_vol', 'taker_quote_vol', 'ignore']
    df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in df.close_time]
    return df







if __name__=="__main__":
    main()