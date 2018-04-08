import numpy as np
from pandas import DataFrame, Series, concat, datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from math import sqrt
import matplotlib.pyplot as plt

from lstm_predictor import difference, timeseries_to_supervised, scale, fit_lstm, invert_scale, inverse_difference, forecast_lstm

# load dataset
#series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

TIMESTEPS = 500

def train(data):

    TIMESTEPS = 500

    # transform data to be stationary
    raw_values = data[4].astype(float)
    diff_values = difference(raw_values, 1)

    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    # split data into train and test-sets
    train, test = supervised_values[0:-TIMESTEPS], supervised_values[-TIMESTEPS:]

    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)

    # fit the model
    lstm_model = fit_lstm(train_scaled, 1, 2, 4)
    print('fitting finished')
    '''# forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)
    print('prediction finished')
    # walk-forward validation on the test data
    predictions = list()
    for i in range(len(test_scaled)):
    	# make one-step forecast
    	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    	yhat = forecast_lstm(lstm_model, 1, X)
    	# invert scaling
    	yhat = invert_scale(scaler, X, yhat)
    	# invert differencing
    	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    	# store forecast
    	predictions.append(yhat)
    	expected = raw_values[len(train) + i + 1]
    	print('Minute=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

    # report performance
    rmse = sqrt(mean_squared_error(raw_values[-TIMESTEPS:], predictions))
    print('Test RMSE: %.3f' % rmse)
    # line plot of observed vs predicted
    plt.plot(raw_values[-TIMESTEPS:])
    plt.plot(predictions)
    plt.show()'''
    return lstm_model

def predict(lstm_model,data) :
     # transform data to be stationary
    print (data.shape)
    raw_values = data[4].astype(float)
    diff_values = difference(raw_values, 1)
    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values
    # split data into train and test-sets
    train, test = supervised_values[0:-TIMESTEPS], supervised_values[-TIMESTEPS:]
    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    predictions = list()
    for i in range(len(test_scaled)):
    	# make one-step forecast
    	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    	yhat = forecast_lstm(lstm_model, 1, X)
    	# invert scaling
    	yhat = invert_scale(scaler, X, yhat)
    	# invert differencing
    	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    	# store forecast
    	predictions.append(yhat)
    	expected = raw_values[len(train) + i + 1]
    	#print('Minute=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

    # report performance
    rmse = sqrt(mean_squared_error(raw_values[-TIMESTEPS:], predictions))
    print('Test RMSE: %.3f' % rmse)
    # line plot of observed vs predicted

    indexes = np.arange(TIMESTEPS*2)
    plt.plot(raw_values[-TIMESTEPS*2:])
    print(raw_values[-TIMESTEPS*2:].shape)
    plt.plot(indexes[TIMESTEPS:], predictions)
    plt.show()
    return predictions