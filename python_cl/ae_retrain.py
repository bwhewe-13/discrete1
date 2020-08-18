#!/usr/bin/env python

import numpy as np
# import discrete1.rogue as r
from discrete1.util import nnets
# import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.layers import Input,Dense
# from tensorflow.keras.models import Model
# from tensorflow.keras import regularizers
import argparse
import os

parser = argparse.ArgumentParser(description='Data Input')
parser.add_argument('-nodes',action='store',dest='en',nargs='+') #Enrichment looking into
parser.add_argument('-data',action='store',dest='data') # Data file name
parser.add_argument('-model',action='store',dest='model')
parser.add_argument('-gpu',action='store',dest='gpu')
usr_input = parser.parse_args()

if usr_input.gpu is None:
    print('No GPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    print('Using GPU')

nodes = [int(jj) for jj in usr_input.en]
file1 = '-'.join([str(jj) for jj in usr_input.en])
mymat = np.load('mydata/ae_model_data/{}.npy'.format(usr_input.data))

split = 0.2
try:
    spatial = np.load('autoencoder/{}/spatialShuffleMat.npy'.format(usr_input.model))
except FileNotFoundError:
    spatial = np.arange(len(mymat))
    np.random.shuffle(spatial)
    np.save('autoencoder/{}/spatialShuffleMat'.format(usr_input.model),spatial)
# Normalize the data
mymat,maxi,mini = nnets.normalize(mymat,verbose=True)
mymat[np.isnan(mymat)] = 0

index = int(len(spatial)*split)
train = mymat[spatial[index:]]; test = mymat[spatial[:index]]

autoencoder = keras.models.load_model('autoencoder/{}/model{}_autoencoder2.h5'.format(usr_input.model,file1))

# encoder = Model(autoencoder.input, autoencoder.layers[-2].output)
# decoder_input = Input(shape=(nodes[-1],))
# decoder = Model(decoder_input, autoencoder.layers[-1](decoder_input))

autoencoder.compile(optimizer='adam',loss='mse')
history = autoencoder.fit(train,train,epochs=10000,validation_data=(test,test))

autoencoder.save('autoencoder/{}/model{}_autoencoder3.h5'.format(usr_input.model,file1))
# encoder.save('autoencoder/{}/model{}_encoder2.h5'.format(usr_input.model,file1))
# decoder.save('autoencoder/{}/model{}_decoder2.h5'.format(usr_input.model,file1))
np.save('autoencoder/{}/loss3_{}'.format(usr_input.model,file1),history.history['loss'])
