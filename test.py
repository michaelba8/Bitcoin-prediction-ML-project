import bitcoin_prediction as bp
import numpy as np
import sklearn.model_selection as ms


def main():
    """
    Tester for the module bitcoin_prediction.py
    the test code is not the most efficient,
    because every function designed to be self-sufficient so there are duplicate actions such as reading the data.

    """
    lr_model,lr_scaler,x_test1,y_test1=bp.create_logistic_regression(debug=True)
    nn_model,nn_scaler,x_test2,y_test2=bp.neural_network(debug=True)
    data = bp.read_data()
    Y = bp.create_Y(data)
    X, scaler = bp.create_X(np.copy(data))
    x_train, x_test, y_train, y_test = ms.train_test_split(X, Y, test_size=0.1, random_state=56)

    print('results:\n')
    bp.custom_accuracy_test(lr_model, x_test1, y_test1, probability=0.62, title='Logistic Regression custom accuracy: ')
    bp.custom_accuracy_test(nn_model, x_test2, y_test2, probability=0.62, title='Neural Network custom accuracy: ')
    print('most significant features: ', bp.most_significant_features(data))
    print('\nC optimisation for logistic regression:')
    bp.best_logistic_reg(x_train,x_test,y_train,y_test)



if __name__=='__main__':
    main()