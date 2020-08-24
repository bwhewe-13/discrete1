#!/usr/bin/env python

import numpy as np
# import discrete1.rogue as r
from discrete1.util import nnets
# import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import argparse, os
from sklearn import model_selection

parser = argparse.ArgumentParser(description='Data Input')
parser.add_argument('-nodes',action='store',dest='en',nargs='+') #Enrichment looking into
parser.add_argument('-data',action='store',dest='data') # Data file name
parser.add_argument('-model',action='store',dest='model')
parser.add_argument('-gpu',action='store',dest='gpu')
parser.add_argument('-source',action='store',dest='source')
usr_input = parser.parse_args()

if usr_input.gpu is None:
    print('No GPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    print('Using GPU')

nodes = [int(jj) for jj in usr_input.en]
file1 = '-'.join([str(jj) for jj in usr_input.en])

if usr_input.source is None:
    label1 = ''
else:
    label1 = '_source'
mymat = np.load('mydata/ae{}_model_data/{}_{}.npy'.format(label1,usr_input.data,usr_input.model))

split = 0.2
try:
    spatial = np.load('autoencoder/{}{}/spatialShuffleMat.npy'.format(usr_input.model,label1))
except FileNotFoundError:
    spatial = np.arange(len(mymat))
    np.random.shuffle(spatial)
    np.save('autoencoder/{}{}/spatialShuffleMat'.format(usr_input.model,label1),spatial)
# Normalize the data
mymat,maxi,mini = nnets.normalize(mymat,verbose=True)
mymat[np.isnan(mymat)] = 0

# index = int(len(spatial)*split)
# train = mymat[spatial[index:]]; test = mymat[spatial[:index]]
train,test = model_selection.train_test_split(mymat,test_size=0.2,random_state=47)

dim = 87
ddims = nodes[:len(nodes)-1][::-1]

input_shape = Input(shape=(dim,))
encoded = Dense(nodes[0],activation='relu')(input_shape) #,kernel_regularizer=regularizers.l2(0.05)
if len(nodes) > 1:
    for ii in range(len(nodes)-1):
        encoded = Dense(nodes[ii+1],activation='relu')(encoded) #,kernel_regularizer=regularizers.l2(0.05)
    for jj in range(len(ddims)):
        if jj == 0:
            decoded = Dense(ddims[jj],activation='relu')(encoded) #,kernel_regularizer=regularizers.l2(0.05)
        else:
            decoded = Dense(ddims[jj],activation='relu')(decoded) #,kernel_regularizer=regularizers.l2(0.05)
    decoded = Dense(dim,activation='sigmoid')(decoded)
else:
    decoded = Dense(dim,activation='sigmoid')(encoded)
# Build Autoencoder
autoencoder = Model(input_shape,decoded)
# Build Encoder
encoder = Model(input_shape,encoded)
# Build Decoder
encoded_input = Input(shape=(nodes[-1],))
decoded = autoencoder.layers[len(nodes)+1](encoded_input)
for ii in range(len(nodes)-1):
    decoded = autoencoder.layers[len(nodes)+2+ii](decoded)
decoder = Model(encoded_input,decoded)
# Compile Autoencoder
autoencoder.compile(optimizer='adam',loss='mse')
autoencoder.summary()
history = autoencoder.fit(train,train,epochs=200,validation_data=(test,test))

autoencoder.save('autoencoder/{}{}/model{}_{}_autoencoder.h5'.format(usr_input.model,label1,file1,usr_input.data))
encoder.save('autoencoder/{}{}/model{}_{}_encoder.h5'.format(usr_input.model,label1,file1,usr_input.data))
decoder.save('autoencoder/{}{}/model{}_{}_decoder.h5'.format(usr_input.model,label1,file1,usr_input.data))
np.save('autoencoder/{}{}/loss_{}_{}'.format(usr_input.model,label1,usr_input.data,file1),history.history['loss'])
np.save('autoencoder/{}{}/val_loss_{}_{}'.format(usr_input.model,label1,usr_input.data,file1),history.history['val_loss'])

# dnn = keras.Sequential()
# dnn.add(keras.layers.Dense(nodes[0],input_shape=(dim,),activation='relu',kernel_regularizer=regularizers.l2(0.05)))
# if len(nodes) > 1:
# 	for ii in range(len(nodes)-1):
# 		dnn.add(keras.layers.Dense(nodes[ii+1],activation='relu',kernel_regularizer=regularizers.l2(0.05)))
# 	for jj in range(len(ddims)):
# 		dnn.add(keras.layers.Dense(ddims[jj],activation='relu',kernel_regularizer=regularizers.l2(0.05)))
# dnn.add(keras.layers.Dense(dim,activation='sigmoid'))

# dnn.summary()

# dnn.compile(optimizer='adam',loss='mse')
# history = dnn.fit(train,train,epochs=1000,validation_data=(test,test))

# dnn.save('autoencoder/{}/model{}.h5'.format(usr_input.data,file1))
# np.save('autoencoder/{}/loss_{}'.format(usr_input.data,file1),history.history['loss'])
# I am not saving validation loss