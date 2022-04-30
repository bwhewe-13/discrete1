#!/usr/bin/env python

from discrete1.utils import nnets
from djinn import djinn

import numpy as np
from tensorflow import keras
import sklearn
from sklearn import model_selection
import argparse
from glob import glob
import os

parser = argparse.ArgumentParser(description='Specific Type of Search')
parser.add_argument('-xs',action='store',dest='xs') # "fission" or "scatter"
parser.add_argument('-label',action='store',dest='label') # "True" if labeled data
parser.add_argument('-model',action='store',dest='model') # type of material (pluto or hdpe)
parser.add_argument('-count',action='store',dest='count') # Variation Number
# parser.add_argument('-data',action='store',dest='data') # The specific problem, "multiplastic," "pluto," "carbon"
usr_input = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def randomize(data, model):
    if model == "pluto":
        idx = np.argwhere(data[:,:,0] != 15.04).flatten()
    elif model == "hdpe":
        idx = np.argwhere(data[:,:,0] == 15.04).flatten()
    else:
        raise ValueError("You Messed Up")
    data = data[:,idx].copy()
    dim = np.arange(data.shape[1])
    np.random.shuffle(dim)
    data = data[:,dim][:150000].copy()
    return data, dim[:150000]

if usr_input.xs == "scatter":
    short_name = "smult"
else:
    short_name = "fmult"

# Load Encoder Models
phi_encoder = keras.models.load_model('autoencoder/{}_{}_high/model_300-150_phi_encoder_{}.h5'.format( \
                    usr_input.model, usr_input.xs, usr_input.count))
xs_encoder = keras.models.load_model('autoencoder/{}_{}_high/model_300-150_{}_encoder_{}.h5'.format( \
                    usr_input.model, usr_input.xs, short_name, usr_input.count))

# Load Data
# if usr_input.xs == "scatter":
#     enrichments = ["05", "10", "15", "20", "25"]
#     data = np.sort(glob("mydata/track_pluto_djinn/enrich_05*"))
#     data = np.hstack([np.load(dat) for dat in data])
#     data, idx = randomize(data, usr_input.model)
#     for enrich in enrichments:
#         part_data = np.sort(glob("mydata/track_pluto_djinn/enrich_{}*".format(enrich)))
#         part_data = np.hstack([np.load(dat) for dat in part_data])
#         part_data, part_idx = randomize(part_data, usr_input.model)
#         data = np.hstack((data, part_data))
#         idx = np.append(idx, part_idx)
#         del part_data, part_idx
# elif usr_input.xs == "fission":
#     data = np.sort(glob("mydata/model_data_djinn/fission_pluto_full*"))
#     data = np.hstack([np.load(add) for add in data])
data = np.sort(glob("mydata/model_data_djinn/{}_{}_full*".format(usr_input.xs, usr_input.model)))
data = np.hstack([np.load(add) for add in data])

# Split the data
labels = data[0,:,0].copy()
X = data[0,:,1:].copy()
Y = data[1,:,1:].copy()
del data

# Normalize Data
if usr_input.xs == "scatter":
    X = X**(1/3)
    Y = Y**(1/3)
elif usr_input.xs == "fission":
    X = nnets.normalize(X)
    Y = nnets.normalize(X)

# Remove undesirables
X[np.isnan(X)] = 0
X[np.isinf(X)] = 0
Y[np.isnan(Y)] = 0
Y[np.isinf(Y)] = 0

# Encode the data
X = phi_encoder.predict(X)
Y = xs_encoder.predict(Y)

# Add label
if usr_input.label is None:
    labeled = "reg"
else:
    X = np.hstack((labels[:,None], X))
    labeled = "label"

num_trees = [1, 2, 3]
num_depth = [2, 3, 4] 

x_train, x_test, y_train, y_test = model_selection.train_test_split( \
                                                    X, Y, test_size=0.2)

for jj in num_trees:
    for ii in num_depth:
        print("ntrees:",jj,"maxdepth:",ii)
        print('='*25)
        z_jj = str(jj).zfill(3)
        z_ii = str(ii).zfill(3)
        modelname="model_{}{}".format(z_jj, z_ii) 
        model_path = "{}_1d/{}_high_{}/djinn_{}/".format(usr_input.xs, \
                        usr_input.model, usr_input.count, labeled)
        ntrees = jj                        # number of trees = number of neural nets in ensemble
        maxdepth = ii                      # max depth of tree -- optimize this for each data set
        dropout_keep = 1.0                 # dropout typically set to 1 for non-Bayesian models

        # initialize the model
        model=djinn.DJINN_Regressor(ntrees,maxdepth,dropout_keep)

        # find optimal settings
        optimal=model.get_hyperparameters(x_train,y_train)
        batchsize=optimal['batch_size']
        learnrate=optimal['learn_rate']
        epochs=np.min((300,optimal['epochs']))
        #epochs=optimal['epochs']

        # train the model with these settings
        model.train(x_train,y_train, epochs=epochs,learn_rate=learnrate, batch_size=batchsize, 
                      display_step=1, save_files=True, model_path=model_path,file_name=modelname, 
                      save_model=True,model_name=modelname)

        m=model.predict(x_test)

        # evaluate results
        with open('{}error/model_ntrees{}_maxdepth{}.txt'.format(model_path,z_jj,z_ii),'a') as f:
            for kk in range(y_test.shape[0]):
                mse=sklearn.metrics.mean_squared_error(y_test[kk],m[kk])
                mabs=sklearn.metrics.mean_absolute_error(y_test[kk],m[kk])
                exvar=sklearn.metrics.explained_variance_score(y_test[kk],m[kk])   
                f.write('MSE '+str(mse)+' M Abs Err '+str(mabs)+' Expl. Var. '+str(exvar)+'\n')
        # close model 
        model.close_model()
