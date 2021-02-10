import bitcoin_prediction as BP
import json
import requests
import pandas as pd
import sys
import datetime as dt
import numpy as np



openTime = 0
openPrice = 1
highPrice = 2
lowPrice = 3
closePrice = 4
volume = 5
closeTime = 6
quoteAV = 7
trades = 8
takerBaseAV = 9
takerQuoteAV = 10
ignored=11

def main():
    df=get_bars()
    data=df.to_numpy()
    data=np.array(data,dtype='float64')
    value_change=0.002
    invest=1000
    model,scaler=BP.neural_network(value_change=value_change,max_minutes=5)
    X=BP.setX(np.copy(data),scaler)
    is_in=False
    tries=0
    wins=0
    long=0
    short=0
    for i in range(data.shape[0]):
        if(not is_in):
            prediction=BP.predict(model,X[[i],:],0.95)
            if(prediction==0):
                continue

            btd=data[i,closePrice]
            dtb=1/btd
            position=invest*dtb
            is_in=True
            print('buy ',str(position),'bitcoin for ',invest,'prediction: ',prediction, 'value: ',btd)
            tries+=1
        else:
            high=data[i,highPrice]/btd
            low=data[i,lowPrice]/btd
            if(high>1+value_change):
                if(prediction==1):
                    invest=position*data[i,highPrice]
                    wins+=1
                    long+=1
                else:
                    invest=position*(btd**2/data[i,highPrice])
                print('out- position: ',str(position),'invest: ',str(invest),'value before/after: ',btd,data[i,highPrice])
                is_in=False
                continue
            if(low<1-value_change):
                if(prediction==-1):
                    invest=position*(btd**2/data[i,lowPrice])
                    print('out- position: ',str(position),'invest: ',str(invest),'value before/after: ',btd,data[i,lowPrice])
                    wins+=1
                    short+=1
                else:
                    invest=position*data[i,lowPrice]
                    print('out- position: ',str(position),'invest: ',str(invest),'value before/after: ',btd,data[i,lowPrice])
                print()
                is_in = False
                continue

    print (invest)
    print('wins: ',wins)
    print('loses: ',tries-wins)
    print('short: ',short)
    print('long: ',long)






def get_bars(symbol="BTCUSDT", interval="1m", limit="1500"):
    url = 'https://fapi.binance.com/fapi/v1/klines' + '?symbol=' + symbol + '&interval=' + interval + '&limit=' + limit
    data = json.loads(requests.get(url).text)
    df = pd.DataFrame(data)
    df.columns = ['open_time',
                  'o', 'h', 'l', 'c', 'v',
                  'close_time', 'qav', 'num_trades',
                  'taker_base_vol', 'taker_quote_vol', 'ignore']
    df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in df.close_time]
    print(df)
    return df

if __name__=='__main__':
    main()