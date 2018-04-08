import numpy as pd
import requests
import json


def loadDataTimestamp(item, start, end):
    #data = pd.read_csv('coinbaseEUR.csv', header=None, error_bad_lines=False)
    #data.index = pd.to_datetime((data.index.values*1e9).astype(int))
    external = 0
    external_adr = '86.64.78.32:30000'
    internal_adr = '10.8.176.101:30000'
    if external == 1:
        r = requests.get('http://'+str(external_adr)+'/api/'+str(item)+'/'+str(start)+'/'+str(end))
    else:
        r = requests.get('http://'+str(internal_adr)+'/api/'+str(item)+'/'+str(start)+'/'+str(end))
    data = pd.array(r.json())
    data_array = []
    for item in data:
        data_sub = []
        data_sub.append(item['id'])
        data_sub.append(item['timestamp'])
        data_sub.append(item['volume'])
        data_sub.append(item['opening'])
        data_sub.append(item['closing'])
        data_sub.append(item['high'])
        data_sub.append(item['low'])
        data_array.append(data_sub)

    #print(str(data_array[0][2]))
    #print(str(data.shape))
    return data_array

def loadAllData(item) :
    external_adr = '86.64.78.32:30000'
    external = 0
    internal_adr = '10.8.176.101:30000'
    if external == 1:
        print ("Requesting server for data by external adress")
        r = requests.get('http://'+str(external_adr)+'/api/'+str(item)+'/all/')
    else:
        print ("Requesting server for data by internal adress")
        r = requests.get('http://'+str(internal_adr)+'/api/'+str(item)+'/all/')
    data = pd.array(r.json())
    data_array = []
    for item in data:
        data_sub = []
        data_sub.append(item['id'])
        data_sub.append(item['timestamp'])
        data_sub.append(item['volume'])
        data_sub.append(item['opening'])
        data_sub.append(item['closing'])
        data_sub.append(item['high'])
        data_sub.append(item['low'])
        data_array.append(data_sub)
    return data_array