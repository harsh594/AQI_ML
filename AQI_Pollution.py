# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 12:40:59 2018

@author: Darpan
"""

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('aqidaily2018(conc).csv')
X = dataset.iloc[:,[3,4,7,8,9]].values
y = dataset.iloc[:,10].values
dataset1 = pd.read_csv('testdata.csv')

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_test=dataset1.iloc[:,:].values

# Part 2 - Now let's make the ANN!
np.random.seed(1337)
# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense,Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))
classifier.add(Dropout(0.25))
# Adding the second hidden layer
classifier.add(Dense(units = 41, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'softplus'))
# Adding the output layer
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))

# Compiling the ANN
classifier.compile(loss='mean_squared_error', optimizer='adadelta')

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size =10, epochs =800,shuffle=False)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = classifier.predict(np.array([[6,22,120,25,10]]))
##ls.append(y_pred)
#Rando forest

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators =90, random_state = 0)
regressor.fit(X_train,y_train)

                                                 
y_pred2 = regressor.predict(X_test)
#y_pred2 = regressor.predict(np.array([[6,22,141,19,10]]))

y_pred2 = np.reshape(y_pred2, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))
k=[]
b=0
for x,y in np.c_[y_pred,y_test]:
    print(x-y)
    if abs(x-y)>20:
        pass
    else:
        b+=1
    
    k.append(x-y)

l=[]
c=0
for x,y in np.c_[y_pred2,y_test]:
    print(x-y)
    if abs(x-y)>20:
        pass
    else:
        c+=1
    
    l.append(x-y)

