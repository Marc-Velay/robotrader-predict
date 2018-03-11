from sklearn.preprocessing import MinMaxScaler
import numpy as np

def getLinear(values) :
    i = 0
    final = []
    j = values[i]
    aux = []
    comp = 0
    print (len(values))
    while i < (len(values)-1) :
        if(checkDist(j,values[i+1])) :
            aux.append(values[i+1])
        else:
            if(len(aux)>30) :
                final.append(aux)
            else:
                comp +=len(aux)
            aux = []
            j = values[i]
        i+=1
    if(len(aux)>30) :
        final.append(aux)
    else:
        comp +=len(aux)
    return (final,comp)

def checkDist(ref, todo) :
    return (not (float(todo[0])>(float(ref[0]) + (float(ref[0])*2)/100) or float(todo[0]) <(float(ref[0]) - (float(ref[0])*2)/100))) and (not (float(todo[1])>(float(ref[1]) + (float(ref[1])*2)/100) or float(todo[1]) <(float(ref[1]) - (float(ref[1])*2)/100)))

def setSvmValues(values) :
    expected = []
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for dataset in values :
        aux = []
        for row in dataset:
            aux.append(row[3])
        expected.append(aux)
    for dataset in expected :
        del(dataset[0])
    print (len(values[0]))
    for i in range(0,len(values)) :
        values[i] = np.delete(values[i],len(values[i])-1,0)
        values[i] = scaler.fit_transform(values[i],expected[i])
    train = values[:int(len(values)-(len(values)*0.3))]
    expTrain = expected[:int(len(expected)-(len(expected)*0.3))]
    check = values[int(len(values)-(len(values)*0.3)):]
    expcheck = expected[int(len(expected)-(len(expected)*0.3)):]
    return train,expTrain,check,expcheck  