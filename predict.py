import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot

from loader import loadData
from linearReg import regression

import time
import pickle
import os.path

data_pkl = 'data_gdax.pkl'

if __name__ == "__main__":
    start_time = time.time()
    if not os.path.isfile(data_pkl):
        data = loadData('GDAX', 1496275200, 1501545600)
        dataT = np.transpose(data)
        with open(data_pkl, 'wb') as fid:
            pickle.dump(dataT, fid)
    else:
        with open(data_pkl, 'rb') as fid:
            dataT = pickle.load(fid)
    #print(data['2015-05']) #select all entries for may 2015
    #print(data.iloc[:,0]) #select the first column, besides index col timestamp
    #rolmean = data.iloc[:,0].rolling(center=False, window=10000).mean()

    #mean = plt.plot(rolmean['2018'], color='red', label='Rolling Mean')


    preds = regression(dataT)
    #plt.plot(dataT[1][len(dataT)-1000:len(dataT)-500], dataT[4][len(dataT)-1000:len(dataT)-500], 'b')
    plt.plot(dataT[1][::200], dataT[4][::200], 'b')
    #plt.plot(dataT[1][len(dataT[1])-499:], preds, 'r')
    print(preds[::10])
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()
    #print(data.transpose()[0])
