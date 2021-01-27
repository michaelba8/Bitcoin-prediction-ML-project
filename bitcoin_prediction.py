import json
import requests
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



def main():
    df=get_bars()
    print(df)
    X=np.float64(df)
    print(X.shape)





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