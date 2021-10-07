#install the packages

import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
from datetime import datetime as dt

#read the data 

df=pd.read_csv(r"/Users/yanyuchen/Dropbox/fivesecbars.csv")
df.head()
df.dtypes

#the epochtime is integer obejct, transfer it to date object 

df['epochtime'] = pd.to_datetime(df['epochtime'],yearfirst=1970, unit='s')
df = df.rename(columns={'epochtime':'date'})

#set date as index of the data, and sort the data by the date 
df = df.set_index('date')
df.sort_values(by=['date'], inplace=True, ascending=True)
df.tail()

#check NAs
df.dropna(axis=0, inplace=True)
df.isna().sum()

#now selecting the first stock, which tickerid=0, and then do the analysis stock by stock. 
s0=df[df["tickerid"]==0]

#plot stock(tickerid=0)
plt.plot(s0['weightedavgprice'])
plt.ylabel('Close prices')
plt.show()

#get the price at lag 1, x(t)~y(t+1)
num = 1
s0['label']=s0['weightedavgprice'].shift(-num)
s0

#since the data is highly correlated with time, to cancel the correlation, use bootstrap to randomly select samples. 
from sklearn.utils import resample

boot0 = resample(s0, replace=True, n_samples=len(s0), random_state=1)
boot0

#data preparation for linear model fit 

#import packages 
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

#predictors set 
boot0.dropna(axis=0, inplace=True)
Data = boot0.drop(['tickerid'], axis=1)
X = Data.values

#standardized the predictive varibles 
X = preprocessing.scale(X)
X = X[:-num]

Target = boot0.label
y = Target.values

print(np.shape(X), np.shape(y))

#set up the ratio of training data and testing data 

size= int(0.7*s0.shape[0])

X_train = X[0:size, :]
y_train=y[0:size]

#X_test, y_test = X[size:,:], y[size:len(boot0)-1]
X_test = X[size:,:]
y_test = y[size:len(boot0)-num]


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

lr = LinearRegression()
lr.fit(X_train, y_train)

#use the linear model of x(t) to predict y(t+1)
X_Predict = X[-num:]
Forecast = lr.predict(X_Predict)
print(Forecast)
#the price of next 5 seconds will be 794.01

#now evaluate the model's RSME
preds=lr.predict(X_test)
rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))
rms


#################################################
#what if inccrease the lag, now lag=10
num = 10
s0['label']=s0['weightedavgprice'].shift(-num)
s0

#since the data is highly correlated with time, to cancel the correlation, use bootstrap to randomly select samples. 
from sklearn.utils import resample

boot0 = resample(s0, replace=True, n_samples=len(s0), random_state=1)
boot0

#data preparation for linear model fit 

#import packages 
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

#predictors set 
boot0.dropna(axis=0, inplace=True)
Data = boot0.drop(['tickerid'], axis=1)
X = Data.values

#standardized the predictive varibles 
X = preprocessing.scale(X)
X = X[:-num]

Target = boot0.label
y = Target.values

print(np.shape(X), np.shape(y))

#set up the ratio of training data and testing data 

size= int(0.7*s0.shape[0])

X_train = X[0:size, :]
y_train=y[0:size]

#X_test, y_test = X[size:,:], y[size:len(boot0)-1]
X_test = X[size:,:]
y_test = y[size:len(boot0)-num]


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

lr = LinearRegression()
lr.fit(X_train, y_train)

#use the linear model of x(t) to predict y(t+1)
X_Predict = X[-num:]
Forecast = lr.predict(X_Predict)
print(Forecast)

#now evaluate the model's RSME
preds=lr.predict(X_test)
rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(preds)),2)))
rms
