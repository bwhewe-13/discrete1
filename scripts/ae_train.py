#!/usr/bin/env python

import numpy as np
from discrete1.util import nnets
# import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import argparse, os, glob
from sklearn import model_selection

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


# mymat = np.load('mydata/ae_model_data/{}_{}.npy'.format(usr_input.data,usr_input.model))
address = np.sort(glob.glob('mydata/ae_model_data/{}_enrich*'.format(usr_input.data)))
mymat = nnets.randomize(address,150000)
print('SHAPE OF DATA',mymat.shape)

# Normalize the data
mymat = nnets.normalize(mymat)
mymat[np.isnan(mymat)] = 0

train,test = model_selection.train_test_split(mymat,test_size=0.2)

dim = 618
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
history = autoencoder.fit(train,train,epochs=100,validation_data=(test,test))

autoencoder.save('autoencoder/{}/model_{}_{}_autoencoder_001.h5'.format(usr_input.model,file1,usr_input.data))
encoder.save('autoencoder/{}/model_{}_{}_encoder_001.h5'.format(usr_input.model,file1,usr_input.data))
decoder.save('autoencoder/{}/model_{}_{}_decoder_001.h5'.format(usr_input.model,file1,usr_input.data))
np.save('autoencoder/{}/loss_{}_{}_001'.format(usr_input.model,usr_input.data,file1),history.history['loss'])
np.save('autoencoder/{}/val_loss_{}_{}_001'.format(usr_input.model,usr_input.data,file1),history.history['val_loss'])

