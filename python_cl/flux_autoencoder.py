#!/usr/bin/env python3

import numpy as np
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
# from tensorflow.keras import backend as K
import argparse
from discrete1.losses import original

parser = argparse.ArgumentParser(description='Dimens')
parser.add_argument('-nn',action='store',dest='nn',nargs='+')
parser.add_argument('-l',action='store',dest='label')
usr_input = parser.parse_args()

if usr_input.label is None:
    print('Unlabeled Data')
    X = np.load('mydata/nn_djinn_1d/unlabeled_phi.npy')
    encode_label = ''.join([str(ii) for ii in usr_input.nn])
    dim = 87
else:
    print('Labeled Data')
    X = np.load('mydata/nn_djinn_1d/labeled_phi.npy')
    encode_label = ''.join([str(ii) for ii in usr_input.nn])+'_enrich'
    dim = 88
print('Model',encode_label)

# Variables
reg_val = 0.05; split = 0.2

spatial = np.load('mydata/nn_djinn_1d/spatialShuffleMat.npy')
index = int(len(X)*split)
# Train and Test Split
x_train = X[spatial[index:]]
x_test = X[spatial[:index]]
# Build autoencoder
autoencoder = keras.Sequential()
autoencoder.add(keras.layers.Dense(usr_input.nn[0],input_shape=(dim,),activation='relu',kernel_regularizer=regularizers.l2(0.05)))
for ii in usr_input.nn[1:]:
    autoencoder.add(keras.layers.Dense(ii,activation='relu',kernel_regularizer=regularizers.l2(reg_val)))
for jj in usr_input.nn[::-1][1:]:
    autoencoder.add(keras.layers.Dense(jj,activation='relu',kernel_regularizer=regularizers.l2(reg_val)))
autoencoder.add(keras.layers.Dense(dim,activation='sigmoid'))
autoencoder.summary()

# Work on loss function
weight = 25
loss = original.loss_params(dim,weight)
autoencoder.compile(optimizer='adam',loss=loss)

# Fit to training data
history = autoencoder.fit(x_train,x_train,epochs=100000,validation_data=(x_test,x_test))
# Save model, loss, validation loss
autoencoder.save('mydata/nn_djinn_1d/model_{}.h5'.format(encode_label))
np.save('mydata/nn_djinn_1d/valLoss_{}'.format(encode_label),history.history['val_loss'])
np.save('mydata/nn_djinn_1d/loss_{}'.format(encode_label),history.history['loss'])
# Save in Dropbox
autoencoder.save('../../Dropbox/nn_djinn/model_{}.h5'.format(encode_label))
np.save('../../Dropbox/nn_djinn/valLoss_{}'.format(encode_label),history.history['val_loss'])
np.save('../../Dropbox/nn_djinn/loss_{}'.format(encode_label),history.history['loss'])

# %%
# nn = [60,40,20,10]
# autoencoder = keras.Sequential()
# autoencoder.add(keras.layers.Dense(nn[0],input_shape=(dim,),activation='relu',kernel_regularizer=regularizers.l2(0.05)))
# for ii in nn[1:]:
#     autoencoder.add(keras.layers.Dense(ii,activation='relu',kernel_regularizer=regularizers.l2(0.05)))
# for jj in nn[::-1][1:]:
#     autoencoder.add(keras.layers.Dense(jj,activation='relu',kernel_regularizer=regularizers.l2(0.05)))
# autoencoder.add(keras.layers.Dense(dim,activation='sigmoid'))
# autoencoder.summary()


# data = np.load('mydata/djinn_true_1d/true_track_scatter_combined.npy')
# X = data[:,:2,:].copy() #check this
# X = np.hstack((X[:,0][:,:1],X[:,1]))

# split = 0.2
# spatial = np.arange(len(data))
# np.random.shuffle(spatial)
# np.save('mydata/nn_djinn_1d/spatialShuffleMat',spatial)
# index = int(len(spatial)*split)

# x_train = X[spatial[index:]]
# x_test = X[spatial[:index]]

# np.save('mydata/nn_djinn_1d/labeled_phi',X)
# np.save('mydata/nn_djinn_1d/unlabeled_phi',data[:,1])

