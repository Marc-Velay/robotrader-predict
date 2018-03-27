import sys
from svm import support_vectors_regression
from linearReg import regression
from lstm import lstm
import time
import numpy as np
import pickle
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from lstm_predictor import difference, timeseries_to_supervised, scale, fit_lstm, invert_scale, inverse_difference, forecast_lstm

lstm_pkl = 'lstm_model.h5'
TIMESTEPS = 500

def redoAlgo(name,dataT):
    start_time = time.time()
    if(len(sys.argv)<=1 or name=="regression") :
        lr = regression(dataT)
        with open(name+".pkl", 'wb') as fid:
            pickle.dump(lr, fid)
    elif (name=="svm"):
        lr = support_vectors_regression(np.transpose(dataT))
        with open(name+".pkl", 'wb') as fid:
            pickle.dump(lr, fid)
    elif (name == "lstm"):
        lr = lstm(dataT)
        lr.save(lstm_pkl)
    else :
        sys.exit("Unknown algorithm")
    sendTTT(time.time()-start_time)
    return lr

def predict(algo,dataset) :
    start_time = time.time()
    if("keras" in str(algo.__class__)):
        var =  calculLSTM(algo,dataset)
        var = np.reshape(var, len(var))
    elif ("svm" in str(algo.__class__)) :
        var =  algo.predict(np.transpose(dataset[3:]))
    else:
        var =  algo.predict(np.array([dataset.astype(float)[4]]).reshape(-1, 1))
    sendTTP(time.time()-start_time)
    return var


def calculLSTM(lstm_model,data) :
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

def sendTTP(time):
    print ("Time to predict", time)
    #TODO: send Time to predict to DB

def sendTTT(time) :
    print ("Time to train : ", time)
    #TODO:Send time to train to DB
