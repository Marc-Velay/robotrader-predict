import sys
from svm import support_vectors_regression
from linearReg import regression
from lstm import lstm
import time
import numpy as np
import pickle
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
    elif ("svm" in str(algo.__class__)) :
        var =  algo.predict(np.transpose(dataset[3:]))
    else:
        var =  algo.predict(np.array([dataset.astype(float)[4]]).reshape(-1, 1))
    sendTTP(time.time()-start_time)
    return var


def calculLSTM(lstm,data) :
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
    return lstm.predict(train_reshaped, batch_size=1)

def sendTTP(time):
    print ("Time to predict", time)
    #TODO: send Time to predict to DB

def sendTTT(time) :
    print ("Time to train : ", time)
    #TODO:Send time to train to DB
    