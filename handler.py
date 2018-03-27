import sys, os, time, pickle
from predict import predict
import loader

data_pkl = 'data_gdax.pkl'

def pickleHandler(algo) :
    picklerick = algo+".pkl"
    if not os.path.isfile(picklerick):
        print("Pickle not found, starting training")
        #TODO : Training algo !
    else:
        with open(picklerick, 'rb') as fid:
            algo = pickle.load(fid)
    print ("Trained algorithm loaded")
    return algo

def getdata(item) :
    #TODO : Modify to set on server
    if not os.path.isfile(data_pkl):
        dataT = []
        #data = loader.loadAllData(item)
        #dataT = np.transpose(data)
        print ("Euuuuh ...Not today ! too bad connection")
        with open(data_pkl, 'wb') as fid:
            pickle.dump(dataT, fid)
    else:
        with open(data_pkl, 'rb') as fid:
            dataT = pickle.load(fid)
    return dataT

##### Main #####

if __name__ == "__main__":
    start_time = time.time()

    if(len(sys.argv)<3):
        sys.exit("NOPE ! Did a mistake when calling")
    algo = pickleHandler(sys.argv[2])
    dataT = getdata(sys.argv[3])

