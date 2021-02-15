This module contains Bitcoin prediction models, using Machine Learning algorithms.
in the module there are 2 prediction models:
1.Logistic Regression
2.Neural Network


All the functions of the module can be found in the file bitcoin_prediction.py
test.py is the tester code of the main methods of the module


Module Bitcoin Prediction, functions:
            custom_accuracy_test(model,x_test,y_test,probability=0.62,title='model accuracy: ')
            most_significant_features()
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


All Rights Reserved,
Creator: Michael Ben Amos
Email: michaelba8@gmail.com

