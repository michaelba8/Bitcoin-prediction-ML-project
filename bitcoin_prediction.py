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
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

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
ignored=11

def main():
    """
    Module Bitcoin Prediction, functions:
            custom_accuracy_test(model,x_test,y_test,probability=0.62,title='model accuracy: ')
            def most_significant_features(data=None)
            neural_network(value_change=0.002,max_minutes=5,debug=False)
            create_logistic_regression(value_change=0.002,max_minutes=5,debug=False)
            create_X(X)
            create_Y(X,value=0.002,max_itter=5)
            setX(X,scaler)
            predict(model,X,percentage=0.5)
            read_data(months=20)
            best_logistic_reg(x_train,x_test,y_train,y_test,noC=False)
            mutual_info(x,y)
    for more details for each function use: function.__doc__
    """
    print(main.__doc__)



def custom_accuracy_test(model,x_test,y_test,probability=0.62,title='model accuracy: '):
    """custom accuracy test, testing only the prediction of the classes 1/-1 with probability as an input (default 62%)
        #attack_ratio: how often the model prediction is 1 or -1 """

    hit=0
    miss=0
    y_test=list(y_test)
    pred_list=[]
    actual_list=[]
    for i in range(len(y_test)):
        t=predict(model,x_test[[i],:],probability)
        if(t==1):
            pred_list.append(1)
            actual_list.append(y_test[i])
            if(y_test[i]==1):
                hit+=1

            else:
                miss+=1
        if(t==-1):
            pred_list.append(-1)
            actual_list.append(y_test[i])
            if (y_test[i] == -1):
                hit += 1
            else:
                miss += 1

    print(title,hit/(hit+miss))
    print('hit: ',hit)
    print('miss: ',miss)
    print('total: ' ,hit+miss)
    print('chances: ',x_test.shape[0])
    print('attack ratio: ',(hit+miss)/x_test.shape[0])
    print()
    confusion_matrix_show(pred_list,actual_list,title)

def most_significant_features(data=None):
    """return a list of the most significant features"""
    features = ['open_time',
               'open', 'high', 'low', 'close', 'volume',
               'close_time', 'qav', 'num_trades',
               'taker_base_vol', 'taker_quote_vol', 'ignore']
    costum_features=['high_ratio','low_ratio','close_ratio']
    if(isinstance(data,np.ndarray)):
        data = np.copy(data)
    else:
        data = read_data(months=10)
    Y=create_Y(data,0.002,5)
    most_significant=[]
    for i in range(len(features)):
        feature=features[i]
        logistic_regression = LogisticRegression(solver='lbfgs', C=10, multi_class='auto', max_iter=300)
        temp=np.array(data[:,[i]])
        x_train, x_test, y_train, y_test = ms.train_test_split(temp, Y, test_size=0.1, random_state=56)
        lg = logistic_regression.fit(x_train, y_train)
        prediction=lg.predict(x_test)
        acc = metrics.accuracy_score(y_test, prediction)
        most_significant.append((feature,acc))

    data[:,closePrice]=data[:,closePrice]/data[:,openPrice]
    data[:,highPrice]=data[:,highPrice]/data[:,openPrice]
    data[:,lowPrice]=data[:,lowPrice]/data[:,openPrice]
    data=np.delete(data,[closeTime,openTime,openPrice,takerQuoteAV,takerBaseAV,quoteAV,volume,ignored,trades],1)
    for i in range(len(costum_features)):
        feature=costum_features[i]
        logistic_regression = LogisticRegression(solver='lbfgs', C=10, multi_class='auto', max_iter=300)
        temp=np.array(data[:,[i]])
        x_train, x_test, y_train, y_test = ms.train_test_split(temp, Y, test_size=0.1, random_state=56)
        lg = logistic_regression.fit(x_train, y_train)
        prediction=lg.predict(x_test)
        t1=prediction[prediction!=0]
        t2=y_test[prediction!=0]
        acc = metrics.accuracy_score(y_test, prediction)
        most_significant.append((feature,acc))
    most_significant.sort(key=lambda tup: tup[1],reverse=True)
    result=[]
    for tup in most_significant:
        result.append(tup[0])
    return result


def neural_network(value_change=0.002,max_minutes=5,debug=False):
    """creating model based on neural network, return the scaler as well
        runtime=1-3 minutes
    """
    data=read_data()
    Y=create_Y(data,value_change,max_minutes)
    X,scaler=create_X(np.copy(data))
    x_train, x_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.1, random_state=56)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-05,hidden_layer_sizes = (14, 3), random_state = 1)
    clf.fit(x_train, y_train)
    if(debug):
        return clf, scaler,x_test,y_test
    return clf,scaler


def create_logistic_regression(value_change=0.002,max_minutes=5,debug=False):
    """creating logistic regression model and return the scaler as well"""
    data=read_data()
    Y=create_Y(data,value_change,max_minutes)
    X,scaler=create_X(np.copy(data))
    x_train, x_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.1, random_state=56)
    logistic_regression = LogisticRegression(solver='lbfgs',C=10,multi_class='auto',max_iter=300)
    lg=logistic_regression.fit(x_train,y_train)
    if (debug):
        return lg, scaler, x_test, y_test
    return lg,scaler

def create_X(X):
    """
    :param X: the raw data
    :return: matrix X set for Machine Learning use
        delete most of the features, because they cause overfitting, and change create 3 new features
        the new features are the ratios between the max,min,open,close values of the specific minute (X[i]).
    """
    X[:,closePrice]=X[:,closePrice]/X[:,openPrice]
    X[:,highPrice]=X[:,highPrice]/X[:,openPrice]
    X[:,lowPrice]=X[:,lowPrice]/X[:,openPrice]
    X=np.delete(X,[closeTime,openTime,openPrice,takerQuoteAV,takerBaseAV,quoteAV,volume,ignored,trades],1)
    #X = np.delete(X, [closeTime, openTime, openPrice, ignored], 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, scaler


def create_Y(X,value=0.002,max_itter=5):
    """
    creating the vector Y,
        classes (Y[i]):
             1: increase of value percents from the #i minute of matrix X
            -1: decrease of value percents from the #i minute of matrix X
             0: no change of value percents from the #i minute of matrix X after #max_itter minutes
    """
    m,n=X.shape
    Y=np.zeros(m)
    for i in range(m):
        y,minutes=define_row_y(X,i,value,max_itter)
        Y[i]=y

    return Y

def define_row_y(X,row,value,max_itter=100000000):
    """helper function"""
    y_val=X[row,closePrice]
    i=row
    itter=0
    while(i<X.shape[0] and itter<max_itter):
        high=X[i,highPrice]   #this line might be changed!!!!!!!!!!!!!!!!!!!
        low=X[i,lowPrice]
        temp=high/y_val
        if(temp>1+value):
            return 1,itter
        temp=low/y_val
        if(temp<1-value):
            return -1,itter
        itter+=1
        i+=1
    return 0,itter

def setX(X,scaler):
    """get X and scaler as an input, and return matrix/vector X ready for use in the models of the library """
    X=np.copy(X)
    X[:,closePrice]=X[:,closePrice]/X[:,openPrice]
    X[:,highPrice]=X[:,highPrice]/X[:,openPrice]
    X[:,lowPrice]=X[:,lowPrice]/X[:,openPrice]
    X=np.delete(X,[closeTime,openTime,openPrice,takerQuoteAV,takerBaseAV,quoteAV,volume,ignored,trades],1)
    #X = np.delete(X, [closeTime, openTime, openPrice, ignored], 1)
    X=scaler.transform(X)
    return X

def predict(model,X,percentage=0.5):
    """prediction with custom probability"""
    if(percentage>=1 or percentage<0.35):
        percentage=0.5
    prediciton=model.predict_proba(X)
    if(X.shape[0]==1):
        if(prediciton[0,0]>percentage):
            return -1
        if(prediciton[0,2]>percentage):
            return 1
        return 0
    pred_list=[]
    for i in range(X.shape[0]):
        if (prediciton[i, 0] > percentage):
            pred_list.append( -1)
        if (prediciton[i, 2] > percentage):
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
    result = np.array(temp).astype("float64")
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

    print("read the data successfully ")
    return result




def get_bars(symbol="BTCUSDT", interval="1m", limit="1500"):
    url = 'https://fapi.binance.com/fapi/v1/klines' + '?symbol=' + symbol + '&interval=' + interval + '&limit=' + limit
    data = json.loads(requests.get(url).text)
    df = pd.DataFrame(data)
    df.columns = ['open_time',
                  'o', 'h', 'l', 'c', 'v',
                  'close_time', 'qav', 'num_trades',
                  'taker_base_vol', 'taker_quote_vol', 'ignore']
    df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in df.close_time]
    return np.array(df,dtype=np.float64)

def confusion_matrix_show(res,y_test,title):
    """creating confusion matrix"""
    acc = metrics.accuracy_score(y_test, res)
    ax = plt.axes()
    ax.set_title(title + str(round(acc, 2)))
    confusion_matrix_ovo = pd.crosstab(np.array(y_test), np.array(res), rownames=['Actual'],
                                       colnames=['Predicted'])
    sn.heatmap(confusion_matrix_ovo, annot=True)
    plt.show()


def best_logistic_reg(x_train,x_test,y_train,y_test,noC=False):
    """a method that return the best c and its accuracy for logistic regression on the validation data
    if noC==True than its check the accuracy for one regression with c=1 (used for greedy)
    ,else (default) it check 7 c's and return the best one"""
    c=10**(-5)
    best_c=c
    max_acc=0
    itter=8
    Cs=[]
    Acc=[]
    if(noC):
        c=1
        itter=1
    for i in range(itter):
        lg = LogisticRegression(solver='lbfgs', C=c, multi_class='auto', max_iter=300)
        lg.fit(x_train, y_train)
        result=lg.predict(x_test)
        accuracy=metrics.accuracy_score(y_test,result)
        Cs.append(c)
        Acc.append(accuracy)
        if(accuracy>=max_acc):
            best_c=c
            max_acc=accuracy
        c*=10
    plt.plot(Cs,Acc)
    plt.xlabel('c value')
    plt.ylabel('accuracy')
    ax = plt.gca()
    ax.set_xscale('log')
    plt.show()
    return best_c,accuracy

def mutual_info(x,y):
    """mutual information a method that return a list that contains the values of the algorithm for each feature by index
     most significant features from matrix X has the highest values using probability methods
      X must be normalised!!!!!!"""
    m, n = x.shape
    y_range=(-1,1)
    y_hist,y_bins=np.histogram(y,bins=3,range=y_range,density=True)
    bins=80
    x_range=(-4,4) ## normal distribution edges (value>4 or value<-4 is very unusual)
    most_imp=np.zeros(n)
    for i in range(n):
        sum=0
        xi_hist, xi_bins = np.histogram(x[:,i], bins=bins, range=x_range,density=True)
        xy_hist,x_bins,y_bins=np.histogram2d(x[:,i],y,bins=[bins,3],range=[x_range,y_range],density=True)
        for j in range(len(xi_bins)-1):
            for k in range(3):
                xy_prob=xy_hist[j,k]
                x_prob=xi_hist[j]
                y_prob=y_hist[k]
                if(x_prob==0 or y_prob==0 or xy_prob==0):
                    continue
                temp=xy_prob/(x_prob*y_prob)
                temp=np.log2(temp)
                temp=abs(temp)
                temp*=xy_prob
                sum+=temp
        most_imp[i]=sum
    sort_list=[]
    for i in range(len(most_imp)):
        index=np.argmax(most_imp)
        tup=(index,most_imp[index])
        sort_list.append(tup)
        most_imp[index]=-1
    return sort_list

if __name__=="__main__":
    main()