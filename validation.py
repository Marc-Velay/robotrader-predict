import sys
from svm import support_vectors_regression
from linearReg import regression
from lstm import lstm
import time
import numpy as np
import pickle

lstm_pkl = 'lstm_model.h5'

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
    #TODO:Should be called for the prediction of all algorithms because should be able to monitor it.
    return

def sendTTT(time) :
    print ("Time to train : ", time)
    #TODO:Send time to train to BDD
    return