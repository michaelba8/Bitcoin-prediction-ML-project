
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection as ms
from sklearn import metrics
from sklearn.datasets import load_digits
import seaborn as sn
import bitcoin_prediction as BP

def main():
    """
    Michael ben amos
    203862263
    EX5-Machine Learning
    """

    X,Y=BP.main()
    indexes = []
    classes = [-1,  1]
    for i in range(Y.shape[0]):
        if (Y[i] in classes):
            indexes.append(i)
    indexes = np.array(indexes)
    Y = Y[indexes]  ## creating Y with only 2 classes 3,6
    X = X[indexes, :]  ## creating new X with 2  classes
    """now we have X,Y that contains only rows with class 3 or 6"""
    for i in range (Y.shape[0]):
        if (Y[i] == classes[0]):
            Y[i]=-1
        else:
            Y[i]=1
    """now the classes y contains are only 1 and -1 instead of the original classes"""

    x_train, x_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.20, random_state=2)
    """train=80% ,test=20% """

    m,n=X.shape
    H=[]
    for i in range(n):
        H.append(best_h_for_feature(x_train[:,[i]],y_train))
    """end of part 1, H contains 64 different h's"""
    success_mat=success_matrix(H,x_train,y_train)

    """ end of part 2 the matrix success_mat contains 64 vectors that each one contains where the h_i is wright or wrong"""
    prediction=H_prediction(H,x_test)
    confusion_matrix_show(prediction,y_test,"H(X) confusion matrix: and accuracy: ")

    """end of part 3 we showed the accuracy and confusion matrix of H(x)"""
    H,I,A=Adaboost(H,x_train,y_train,10)
    prediction=weighted_H_prediction(H,x_test,A,I)
    confusion_matrix_show(prediction,y_test,"H(x) using Adaboost confusion matrix and accuracy:")



def Adaboost(H_pool,X,Y,itterations=10):
    """Adaboost algorithm """
    m,n=X.shape
    W=np.ones(X.shape[0])*(1/m)
    H=[]
    itter=0
    alpha_t=[]
    indexes=[]
    while(itter<itterations):
        best_index,error_vec=get_h_from_pool(H_pool,X,Y,W)
        H.append(H_pool[best_index])
        indexes.append(best_index)
        epsilon=error_vec[best_index]
        alpha=np.log((1-epsilon)/epsilon)
        alpha/=2
        alpha_t.append(alpha)
        hit_vector=success_vector(H_pool[best_index],X[:,[best_index]],Y)
        new_W=fix_weights(hit_vector,W,epsilon)
        if(np.array_equal(W,new_W)):
            break
        else:
            W=new_W
        itter+=1
    return H,indexes,alpha_t


def fix_weights(vec,W,epsilon):
    """function that fixes the weight vector W"""
    m=vec.shape[0]
    next_W=np.zeros(m)
    for i in range(m):
        if (vec[i]==1):
            next_W[i]=0.5*(W[i]*(1/(1-epsilon)))
        else:
            next_W[i]=0.5*(W[i]*(1/epsilon))
    return next_W

def get_h_from_pool(H_pool,X,Y,W):
    """function that pick the best h(x) from the pool depending on weights"""
    min_error=2
    best_index=0
    err_vec=[]

    for i, h in enumerate(H_pool):
        prediction = h_prediction(h, X[:, [i]])
        err = error_sum(Y, prediction,W)
        err_vec.append(err)
        if(err<min_error):
            min_error=err
            best_index=i
    return best_index,err_vec

def error_sum(y,prediction,W):
    """function that build the vector Epsilon (error vector) for adaboost"""
    sum=0
    for i in range(y.shape[0]):
        if(y[i]!=prediction[i]):
            sum+=W[i]
    return sum

def weighted_H_prediction(H,X,Alpha,fetures_indexes):
    """function that predict using the Adaboost model"""
    n=len(fetures_indexes)
    m=X.shape[0]
    predictions=[]
    for i in range(n):
        h=H[i]
        prediction=h_prediction(h,X[:,[fetures_indexes[i]]])
        prediction*=Alpha[i]
        predictions.append(prediction)
    final_prediction=np.zeros(m)
    for i in range(n):
        final_prediction+=+predictions[i]
    final_prediction=np.where(final_prediction>0,1,-1)
    return final_prediction







def H_prediction(H,X):
    """function that predict using unweighted H(x)"""
    m,n=X.shape
    prediction_mat=h_prediction(H[0],X[:,[0]])
    for i in range(1,n,1):
        temp=h_prediction(H[i],X[:,[i]])
        prediction_mat=np.vstack((prediction_mat,temp))
    prediction_mat=prediction_mat.transpose()
    prediction=[]
    for i in range(m):
        sum=np.sum(prediction_mat[[i],:])
        if sum>0:
            prediction.append(1)
        else:
            prediction.append(-1)
    return prediction


def success_matrix(H,X,Y):
    """succes matrix"""
    m,n=X.shape
    success_mat = success_vector(H[0], X[:, [0]], Y)
    for i in range(1, n, 1):
        vec = success_vector(H[i], X[:, [i]], Y)
        success_mat = np.vstack((success_mat, vec))
    success_mat = success_mat.transpose()
    return success_mat

def success_vector(h,X,Y):
    """succes vector for h(x) hit=1 miss =-1 size=(m,1)"""
    prediction=h_prediction(h,X)
    s_vector=np.array(prediction)
    for i in range(s_vector.shape[0]):
        if(prediction[i]==Y[i]):
            s_vector[i]=1
        else:
            s_vector[i]=-1
    return s_vector

def h_prediction(h,X):
    """prediction for single h(x)"""
    th=h[0]
    direction=h[1]
    prediction=np.zeros(X.shape[0])
    if(direction>0):
        pos_indexes=np.where(X[:,0]>th)
        neg_indexes=np.where(X[:,0]<=th)
    else:
        pos_indexes = np.where(X[:, 0] <= th)
        neg_indexes = np.where(X[:, 0] > th)
    prediction[pos_indexes]=1
    prediction[neg_indexes]=-1
    return prediction
def best_h_for_feature(X,Y):
    """building stump h(x) using averges"""
    pos_indexes=np.where(Y>0)
    neg_indexes=np.where(Y<0)
    pos_avg=np.sum(X[pos_indexes,0])
    neg_avg=np.sum(X[neg_indexes,0])
    pos_avg/=pos_indexes[0].shape[0]
    neg_avg/=neg_indexes[0].shape[0]
    if(pos_avg<neg_avg):
        direction=-1
    else:
        direction=1
    th=(pos_avg+neg_avg)/2
    best_h=(th,direction)
    return best_h

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