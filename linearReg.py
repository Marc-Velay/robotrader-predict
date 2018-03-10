import numpy as np
from numpy import transpose
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error


def regression(data):
    train, test = data[1:len(data)-500], data[len(data)-500:]

    ir = IsotonicRegression()
    print(train.shape)
    y_ = ir.fit_transform(train[0], train[1])

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

    return predictions
