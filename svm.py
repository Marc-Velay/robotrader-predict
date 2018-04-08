from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
import filtering as pp
import sys

TIMESTEPS = 500

def getData(values) :
    print(np.array(values).shape)
    aux = []
    for i in range(0,len(values)) :
        aux.append(values[i][3:])
    values = aux
    values,nbdata = pp.getLinear(values)
    print(np.array(values).shape)
    return values

def trainSVM(svm,train,expTrain):
    support = [[1,2,3,4]]
    support = np.array(support)
    for i in range(0,len(train)) :
        svm = svm.fit(train[i], expTrain[i])
        support = np.concatenate([support,svm.support_vectors_],axis=0)
    np.delete(support,0,0)
    return support

def evaluate(svm,check,expcheck) :
    score = 0
    print (len(check), len(expcheck))
    for i in range(0, len(check)) :
        score += svm.score(check[i], expcheck[i])
    return score

def predict(svm, array) :
    array = np.transpose(array[3:])
    pred = svm.predict(array)
    indexes = np.arange(TIMESTEPS*2)
    plt.plot(array[-TIMESTEPS*2:])
    print(array[-TIMESTEPS*2:].shape)
    plt.plot(indexes[TIMESTEPS:], pred[:500])
    plt.show()
    return array

def showData(svm,check,expcheck):
    array = predict(svm,check[len(check)-1])
    dates = list(range(1, len(array)+1))
    print (len(array)+1, len(dates))
    plt.plot(dates, array, color='cornflowerblue', lw=2, label='Polynomial model')
    plt.plot(dates, expcheck[len(check)-1], color='darkorange', label='data')

"""
if __name__ == "__main__":
    svr_rbf = SVR(kernel='rbf', C=1e3,gamma=0.1,epsilon=0.01 ,degree=15,cache_size=1000,verbose=True)
    svr_sig = SVR(kernel='sigmoid', C=0.5,epsilon=0.01 ,degree=15,cache_size=1000,verbose=True)
    svr_poly = SVR(kernel='poly', C=1e3,epsilon=0.01 ,degree=5,cache_size=1000,verbose=True)
    print (svr_sig)
    values = getData()
    train,expTrain,check,expcheck = pp.setSvmValues(values)
    #svr_poly.support_vectors_ = trainSVM(svr_poly,train,expTrain)
    #svr_rbf.support_vectors_ = trainSVM(svr_rbf,train,expTrain)
    svr_sig.support_vectors_ = trainSVM(svr_sig,train,expTrain)
    #print (evaluate(svr_sig,check,expcheck))
    #print (evaluate(svr_poly,check,expcheck))
    #print (evaluate(svr_rbf,check,expcheck))
    #showData(svr_poly,check,expcheck)
    showData(svr_sig,check,expcheck)
    #showData(svr_rbf,check,expcheck)
"""

def fit(data,kernel="poly") :
    if (kernel == "rbf") :
        svm = SVR(kernel='rbf', C=1e3,gamma=0.1,epsilon=0.01 ,degree=15,cache_size=1000,verbose=True)
    elif (kernel == "poly") :
        svm = SVR(kernel='poly', C=1e3,gamma=0.1,epsilon=0.01 ,degree=15,cache_size=1000,verbose=True)
    elif (kernel == "sigmoid"):
        svm = SVR(kernel='sigmoid', C=1e3,gamma=0.1,epsilon=0.01 ,degree=15,cache_size=1000,verbose=True)
    else:
        sys.exit("Argument denied")
    data = np.transpose(data)
    values = getData(data)
    train,expTrain = pp.setToTrain(values)
    svm.support_vectors_ = trainSVM(svm,train,expTrain)
    return svm

def getDefaultRBF() :
    return SVR(kernel='rbf', C=1e3,gamma=0.1,epsilon=0.01 ,degree=15,cache_size=1000,verbose=True)

def getDefaultPoly():
    return SVR(kernel='poly', C=1e3,gamma=0.1,epsilon=0.01 ,degree=15,cache_size=1000,verbose=True)

def getDefaultSigmoid():
    return SVR(kernel='sigmoid', C=1e3,gamma=0.1,epsilon=0.01 ,degree=15,cache_size=1000,verbose=True)
