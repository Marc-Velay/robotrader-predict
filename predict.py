import pandas as pd
import numpy as np
#import matplotlib.pylab as plt
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
#matplotlib inline
#from matplotlib.pylab import rcParams

from loader import loadData
from linearReg import regression

#rcParams['figure.figsize'] = 15, 6

data = loadData('GDAX', 1496275200, 1501545600)
#print(data['2015-05']) #select all entries for may 2015
#print(data.iloc[:,0]) #select the first column, besides index col timestamp
'''rolmean = data.iloc[:,0].rolling(center=False, window=10000).mean()

mean = plt.plot(rolmean['2018'], color='red', label='Rolling Mean')
orig = plt.plot(data['2018'].iloc[:,0])

lag_plot(data['2018'].iloc[:,0])

preds = regression(data.iloc[:,0])
plt(data[len(data)-500:])
plt(preds, color="red")
plt.show()'''
#print(data.transpose()[0])
