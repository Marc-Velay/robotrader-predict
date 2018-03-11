import sys


def redoAlgo(name):
    start_time = time.time()
    if(len(sys.argv)<=1 or name=="regression") :
        lr = regression(dataT)
    elif (name=="svm"):
        lr = support_vectors_regression(np.transpose(dataT))
    else :
        sys.exit("Unknown algorithm")
    with open(name+".pkl", 'wb') as fid:
            pickle.dump(lr, fid)
    sendTTT(time.time()-start_time)
    return lr

def sendTTT(time):
    #TODO:Send time to train to BDD