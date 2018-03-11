# pip install numpy, scipy, scikit-learn
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv

values = []
expected = []
with open('GAS.CMDUSD_Candlestick_1_m_BID_20.11.2015-18.11.2017.csv', 'rb') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     for row in spamreader:
         aux = []
         aux.append(float(row[1]))
         aux.append(float(row[2]))
         aux.append(float(row[3]))
         aux.append(float(row[4]))
         aux.append(float(row[5]))
         values.append(aux)

#for data in result:
#    mean = (row[3] + row[2])/2

#train, expTrain = make_regression(n_features=4, random_state=0)
scaler = StandardScaler()
scaler.fit(values)
values = scaler.transform(values)

for row in values:
    aux = row[3]
    expected.append(aux)
del(expected[0])
values = np.delete(values,len(values)-1,0)
print len(values),len(expected),values[len(values)-1][3], expected[len(expected)-1]
train = values[::1000000]
expTrain = expected[::1000000]

check = values[1000000::]
expcheck = expected[1000000::]
svr_rbf = SVR(kernel='rbf', C=1e3,gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=5)
y_rbf = svr_rbf.fit(train, expTrain)
y_lin = svr_lin.fit(train, expTrain)
y_poly = svr_poly.fit(train, expTrain)

print y_lin.score(check, expcheck)*100
print y_poly.score(check, expcheck)*100
print y_rbf.score(check, expcheck)*100

"""
print(regr.coef_)
print(regr.intercept_)
print(regr.predict([[0, 0, 0, 0]]))"""
