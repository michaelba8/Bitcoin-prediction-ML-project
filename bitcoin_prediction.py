import json
import requests
import pandas as pd
import datetime as dt
from sklearn import metrics
import numpy as np
import csv
import sklearn.model_selection as ms
from sklearn.linear_model import LogisticRegression
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression

"""constant define"""
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


def main():
    data=read_data()
    print(data.shape)

    Y=create_Y(data,0.002,5)
    #X=np.hstack((data,minutes_to_change))
    X=create_X(data)
    x_train, x_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.1, random_state=21)
    logistic_regression = LogisticRegression(solver='lbfgs',C=50,multi_class='auto',max_iter=300)
    lg=logistic_regression.fit(x_train,y_train)

    prediction=lg.predict(x_test)
    confusion_matrix_show(prediction,y_test,"prediction accuracy: ")
    
    hit=0
    miss=0
    y_test=list(y_test)
    for i in range(len(y_test)):
        t=predict(lg,x_test[[i],:],0.95)
        if(t==1):
            if(y_test[i]==1):
                hit+=1
            else:
                miss+=1
        if(t==-1):
            if (y_test[i] == -1):
                hit += 1
            else:
                miss += 1

    print('accuracy: ',hit/(hit+miss))
    print('hit: ',hit)
    print('miss: ',miss)
    print('total: ' ,hit+miss)
    print('chances: ',x_test.shape[0])
    print('attack ratio: ',(hit+miss)/x_test.shape[0])

def create_X(X):
    X[:,closePrice]=X[:,closePrice]/X[:,openPrice]
    X[:,highPrice]=X[:,highPrice]/X[:,openPrice]
    X[:,lowPrice]=X[:,lowPrice]/X[:,openPrice]
#    X[:,openTime]=np.roll(X[:,highPrice],1)
 #   X[:, closeTime] = np.roll(X[:, lowPrice], 1)

    X=np.delete(X,[closeTime,openTime,openPrice,takerQuoteAV,takerBaseAV,quoteAV,volume],1)
    X=scale(X,axis=0)
    print(X.shape)
    return X

def create_Y(X,value,max_itter=100000000,logistic=True):
    m,n=X.shape
    Y=np.zeros(m)
    for i in range(m):
        y,minutes=define_row_y(X,i,value,max_itter)
        if(logistic):
            Y[i]=y
        else:
            Y[i]=y*(np.exp(1/(minutes+1))-1)
    return Y

def define_row_y(X,row,value,max_itter=100000000):
    y_val=X[row,openPrice]
    i=row
    itter=0
    while(i<X.shape[0] and itter<max_itter):
        high=X[i,highPrice]   #this line might be changed!!!!!!!!!!!!!!!!!!!
        low=X[i,lowPrice]
        temp=high/y_val
        if(temp>1+value):
            #print("row: "+str(row)+ " value changed in: "+str(i)+" 1")
            return 1,itter
        temp=low/y_val
        if(temp<1-value):
            #print("row: " + str(row) + " value changed in: " + str(i) + " -1")
            return -1,itter
        itter+=1
        i+=1
    return 0,itter


def predict(model,X,percentage=0.5):
    prediciton=model.predict_proba(X)
    if(X.shape[0]==1):
        if(prediciton[0,0]>percentage):
            return -1
        if(prediciton[0,2]>percentage):
            return 1
        return 0
    pred_list=[]
    for i in range(X.shape[0]):
        if (prediciton[0, 0] > percentage):
            pred_list.append( -1)
        if (prediciton[0, 2] > percentage):
            pred_list.append( 1)
        else:
            pred_list.append(0)
    return pred_list

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




def confusion_matrix_show(res,y_test,title):
    """creating confusion matrix"""
    acc = metrics.accuracy_score(y_test, res)
    ax = plt.axes()
    ax.set_title(title + str(round(acc, 2)))
    confusion_matrix_ovo = pd.crosstab(np.array(y_test), np.array(res), rownames=['Actual'],
                                       colnames=['Predicted'])
    sn.heatmap(confusion_matrix_ovo, annot=True)
    plt.show()


if __name__=="__main__":
    main()