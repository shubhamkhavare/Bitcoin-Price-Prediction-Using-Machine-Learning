from math import pi
import pandas as pd 
import numpy as np
from sklearn import svm 

df = pd.read_csv('BTC-USD.csv')

df.drop(['Open'],1,inplace=True)
df.drop(['Date'],1,inplace=True)
df.drop(['High'],1,inplace=True)
df.drop(['Low'],1,inplace=True)
df.drop(['Volume'],1,inplace=True)
df.drop(['Adj Close'],1,inplace=True)

prediction_days = 30
df['Prediction'] = df[['Close']].shift(-prediction_days)

X = np.array(df.drop(['Prediction'],1))
X = X[:len(df)-prediction_days]
print(X)

y = np.array(df['Prediction'])
y = y[:-prediction_days]
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

prediction_days_array = np.array(df.drop(['Prediction'],1))[-prediction_days:]
print(prediction_days_array)

from sklearn.svm import SVR
svr_rbf = SVR(kernel ='rbf', C = 1e3, gamma = 0.00001)
svr_rbf.fit(x_train, y_train)

svr_rbf_confidence = svr_rbf.score(x_test, y_test)
print('svr_rbf accuracy : ', svr_rbf_confidence)

svm_prediction = svr_rbf.predict(x_test)
print(svm_prediction)
print()
print(y_test)

# Predicting the price for next 30 days in USD
svm_prediction = svr_rbf.predict(prediction_days_array)
print(svm_prediction)
print()

# The actual price of Bitcoin for last 30 days in USD
print(df.tail(prediction_days))


import pickle
pickle.dump(svr_rbf,open('model.pkl', 'wb'))

