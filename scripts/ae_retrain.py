#!/usr/bin/env python

import numpy as np
from discrete1.util import nnets
from tensorflow import keras
import argparse, os, glob
from sklearn import model_selection
from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Input,concatenate


parser = argparse.ArgumentParser(description='Data Input')
parser.add_argument('-nodes',action='store',dest='en',nargs='+') #Enrichment looking into
parser.add_argument('-data',action='store',dest='data') # phi, smult, or fmult (values)
parser.add_argument('-model',action='store',dest='model') # type of material
parser.add_argument('-count',action='store',dest='count') # iteration to use
parser.add_argument('-gpu',action='store',dest='gpu')
usr_input = parser.parse_args()

if usr_input.gpu is None:
    print('No GPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    print('Using GPU')

nodes = [int(jj) for jj in usr_input.en]
file1 = '-'.join([str(jj) for jj in usr_input.en])

address = np.sort(glob.glob('mydata/ae_model_data/{}_enrich*'.format(usr_input.data)))
mymat = nnets.randomize(address,150000); dim = 618

# mymat = np.load('mydata/ae_model_data/{}_{}.npy'.format(usr_input.data,usr_input.model)); dim = 87
# Normalize the data
mymat = nnets.normalize(mymat)
mymat[np.isnan(mymat)] = 0

train,test = model_selection.train_test_split(mymat,test_size=0.2)

prev_label = str(int(usr_input.count) - 1).zfill(3)
curr_label = str(usr_input.count).zfill(3)
print('Training Iteration {} from Iteration {}'.format(curr_label,prev_label))

# autoencoder = keras.models.load_model('autoencoder/{}/model_{}_{}_autoencoder_{}.h5'.format(usr_input.model,file1,usr_input.data,prev_label))
encoder = keras.models.load_model('autoencoder/{}/model_{}_{}_encoder_{}.h5'.format(usr_input.model,file1,usr_input.data,prev_label))
decoder = keras.models.load_model('autoencoder/{}/model_{}_{}_decoder_{}.h5'.format(usr_input.model,file1,usr_input.data,prev_label))


input_shape = Input(shape=(dim,))
autoencoder = Model(input_shape,decoder(encoder(input_shape)))


autoencoder.compile(optimizer='adam',loss='mse')
autoencoder.summary()
history = autoencoder.fit(train,train,epochs=100,validation_data=(test,test))

autoencoder.save('autoencoder/{}/model_{}_{}_autoencoder_{}.h5'.format(usr_input.model,file1,usr_input.data,curr_label))
encoder.save('autoencoder/{}/model_{}_{}_encoder_{}.h5'.format(usr_input.model,file1,usr_input.data,curr_label))
decoder.save('autoencoder/{}/model_{}_{}_decoder_{}.h5'.format(usr_input.model,file1,usr_input.data,curr_label))
np.save('autoencoder/{}/loss_{}_{}_{}'.format(usr_input.model,file1,usr_input.data,curr_label),history.history['loss'])
np.save('autoencoder/{}/val_loss_{}_{}_{}'.format(usr_input.model,file1,usr_input.data,curr_label),history.history['val_loss'])
