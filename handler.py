import sys, os, time, pickle
from predict import predict
import loader
import numpy as np
from keras.models import load_model

data_pkl = 'data_gdax.pkl'

def pickleHandler(algo) :
    picklerick = algo+".pkl"
    if not os.path.isfile(picklerick):
        if os.path.isfile(algo+".h5"):
            print("Keras found, try opening with keras_models")
            algo = load_model(algo+".h5")
        else :
            print("Pickle not found, try to dynamically import")
            algo = __import__(algo)
    else:
        with open(picklerick, 'rb') as fid:
            algo = pickle.load(fid)
    print ("Trained algorithm loaded")
    return algo

def getdata(item) :
    #TODO : Modify to set on server
    if not os.path.isfile(data_pkl):
        dataT = []
        data = loader.loadDataTimestamp('gdax', 1514764800, 1522151479)
        #data = loader.loadAllData(item)
        dataT = np.transpose(data)
        print ("Data downloaded")
        with open(data_pkl, 'wb') as fid:
            pickle.dump(dataT, fid)
    else:
        with open(data_pkl, 'rb') as fid:
            dataT = pickle.load(fid)
    return dataT

def train(algo,data) :
    algo = algo.fit(data)
    if("keras" in str(algo.__class__)):
        print("Keras found, try saving with keras_models")
        algo.save(sys.argv[2]+".h5")
    else:
        with open(sys.argv[2]+".pkl", 'wb') as fid:
            pickle.dump(lr, fid)
    print ("Saved")
    return algo
##### Main #####

if __name__ == "__main__":
    

    if(len(sys.argv)<3):
        sys.exit("NOPE ! Did a mistake when calling")
    algo = pickleHandler(sys.argv[2])
    dataT = getdata(sys.argv[3])
    start_time = time.time()
    print (str(algo.__class__))
    # Guess what to do 
    if(sys.argv[1] == "train") :
        print ("Training start")
        train(algo,dataT)
    elif(sys.argv[1] == "pred") :
        print ("Predictions start")
        predict(algo,dataT)
    elif (sys.argv[1] == "test"):
        print ("Starting tests")
        #TODO : test
    else :
        print ("Unknown command, did you make a mistake ?")
