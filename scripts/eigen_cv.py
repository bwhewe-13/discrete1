#!/usr/bin/env python

import numpy as np
from discrete1.util import nnets
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import argparse, sys
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

parser = argparse.ArgumentParser(description='Data Input')
parser.add_argument('-model',action='store',dest='model')
usr_input = parser.parse_args()

mymat = np.load('mydata/ae_model_data/{}_carbon.npy'.format(usr_input.model))
mymat = nnets.normalize(mymat)
mymat[np.isnan(mymat)] = 0

train,test = model_selection.train_test_split(mymat,test_size=0.2) #,random_state=47)

dim = 87
node_label = '40-20'

def create_model(dropout_rate=0.0):
    model = Sequential()
    model.add(Dense(40,input_shape=(dim,),activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(20,activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(40,activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dim,activation='sigmoid'))
    model.summary()
    model.compile(optimizer='adam',loss='mse',metrics=["accuracy"])
    return model

model = KerasClassifier(build_fn=create_model)

# batch_size = [10,20,30,40]
# epochs = [50,100,150,200,250]
# dropout_rate = [0.0,0.1,0.2,0.3]

batch_size = [20,30,40]
epochs = [100,200]
dropout_rate = [0.0,0.1,0.2]
param_grid = dict(batch_size=batch_size,epochs=epochs,dropout_rate=dropout_rate)
grid = GridSearchCV(estimator=model,param_grid=param_grid,cv=3)
grid_result = grid.fit(train,train)


with open('nn_{}_results_{}.txt'.format(usr_input.model,node_label), 'w') as f:
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_),file=f)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param),file=f)