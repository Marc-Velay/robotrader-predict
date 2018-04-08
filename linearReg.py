import numpy as np
from numpy import transpose, reshape
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error


def fit(data):
    len_data = len(data[0])

    train_x, test_x = data[1][0:len_data-501], data[4][len_data-501:]
    train_y, test_y = data[4][0:len_data-499], data[4][len_data-499:]


    lr = LinearRegression()
    lr.fit(train_x.astype(float).reshape(-1,1), train_y.astype(float).reshape(-1,1))
    #ir = IsotonicRegression()
    #ir = ir.fit(train_x, train_y)
    '''model = AR(train)
    model_fit = model.fit()
    print('Lag: %s' % model_fit.k_ar)
    print('Coefficients: %s' % model_fit.params)
    # make predictions
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    for i in range(len(predictions)):
    	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)'''

    return lr
