#Will be removed, shouldn't be used anymore
"""import numpy as np
from numpy import reshape
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot

from loader import loadDataTimestamp
#from linearReg import regression
from lstm import train as lstm

from sklearn.linear_model import LinearRegression
from filtering import scale
#from svm import support_vectors_regression
from  displayDebug import display
from validation import redoAlgo#, predict
import time
import pickle
import os.path
import h5py
import sys
#from keras.models import load_model
from pandas import DataFrame, Series, concat, datetime
#from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
from math import sqrt



data_pkl = 'data_gdax.pkl'
lstm_pkl = 'lstm.h5'

def loadAlgo(name,dataT):
    fullname = name + ".pkl"
    if(name == "lstm"):
        if os.path.isfile(lstm_pkl):
            return load_model(lstm_pkl)
    if not os.path.isfile(fullname):
        return redoAlgo(name,dataT)
    else :
        with open(fullname, 'rb') as fid:
            lr = pickle.load(fid)
        return lr

def predicti(algo,dataT):
    print (str(algo.__class__))
    algo.predict(algo,dataT)[-1000:]
    #predictions = predict(algo,dataT)[-1000:]
    #print (predictions)
    #display(dataT[4][-500:].astype(float),predictions)

if __name__ == "__main__":
    start_time = time.time()
    if not os.path.isfile(data_pkl):
        data = loadDataTimestamp('gdax', 1514764800, 1522151479)
        dataT = np.transpose(data)
        with open(data_pkl, 'wb') as fid:
            pickle.dump(dataT, fid)
    else:
        with open(data_pkl, 'rb') as fid:
            dataT = pickle.load(fid)
    print (dataT.shape)
    #dataT = scale(dataT)
    len_data = len(dataT[1])

    #print(data['2015-05']) #select all entries for may 2015
    #print(data.iloc[:,0]) #select the first column, besides index col timestamp
    #rolmean = data.iloc[:,0].rolling(center=False, window=10000).mean()

    #mean = plt.plot(rolmean['2018'], color='red', label='Rolling Mean')
    if(len(sys.argv)<=1):
        algo = loadAlgo("regression",dataT)
    elif(len(sys.argv)>2) :
        algo = redoAlgo(sys.argv[1],dataT)
    else :
        algo = loadAlgo(sys.argv[1],dataT)

    print (algo)
    predictions = predict(algo,dataT)[-1000:]
    #print (predictions)
    #display(dataT[4][-500:].astype(float),predictions)
    '''lr = regression(dataT)
    #plt.plot(dataT[1][100::200], dataT[4][100::200], 'b')
    #plt.plot(dataT[4][::200], dataT[4][1::200], '.')

    step = len(dataT[1])/500
    #pred_x, pred_y = dataT[1][len_data-500:], dataT[4][len_data-500:]
    pre_pred_x, pre_pred_y = dataT[1], dataT[4]
    preds = lr.predict(pre_pred_x.astype(float).reshape(-1,1))
    preds = np.asarray(preds)

    plt.plot(pre_pred_x.astype(float).reshape(-1,1), pre_pred_y.astype(float).reshape(-1,1), 'b')
    plt.plot(pre_pred_x.astype(float).reshape(-1,1), preds.astype(float).reshape(-1,1), 'r')
    print(dataT[1][0], dataT[1][1])
    '''
    print("--- %s seconds ---" % (time.time() - start_time))
    #plt.show()
    #print(data.transpose()[0])
"""