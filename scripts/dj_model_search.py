#!/usr/bin/env python

import numpy as np
from djinn import djinn
import sklearn
from sklearn import model_selection
import argparse
import glob, os

parser = argparse.ArgumentParser(description='Specific Type of Search')
parser.add_argument('-xs',action='store',dest='xs')                      # Can be either "fission" or "scatter"
parser.add_argument('-label',action='store',dest='label')                # Include -label "True" if input is to be labeled data
parser.add_argument('-model',action='store',dest='model')                # Name of DJINN model to be saved
parser.add_argument('-data',action='store',dest='data')                  # The specific problem, "multiplastic," "pluto," "carbon"
parser.add_argument('-gpu',action='store',dest='gpu')                    # Include if using Tensorflow GPU
usr_input = parser.parse_args()

if usr_input.gpu is None:
    print('No GPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    print('Using GPU')

if usr_input.xs == 'scatter':
    print('Scattering')
    file1 = 'scatter'
elif usr_input.xs == 'fission':
    print('Fission')
    file1 = 'fission'
else:
    file1 = usr_input.xs
    
file2 = usr_input.model+'/'
mymat = np.load('mydata/model_data_djinn/{}_{}_data.npy'.format(usr_input.data,file1))

print(mymat.shape,'shape')
if usr_input.label is None:
    X = mymat[0,:,1:].copy()
    Y = mymat[1,:,1:].copy()
    file3 = '_reg/'
else:
    print('Labeled Data')
    X = mymat[0].copy()
    Y = mymat[1,:,1:].copy() # Remove labels
    file3 = '_label/'

num_trees = [1,2,3]
num_depth = [2,3,4] 

x_train,x_test,y_train,y_test = model_selection.train_test_split(X,Y,test_size=0.2)

for jj in num_trees:
    for ii in num_depth:
        print("ntrees:",jj,"maxdepth:",ii)
        print('='*25)
        z_jj = str(jj).zfill(3); z_ii = str(ii).zfill(3)
        modelname="model_{}{}".format(z_jj,z_ii) 
        model_path = '{}_1d/{}djinn{}'.format(file1,file2,file3)
        ntrees=jj                        # number of trees = number of neural nets in ensemble
        maxdepth=ii                      # max depth of tree -- optimize this for each data set
        dropout_keep=1.0                 # dropout typically set to 1 for non-Bayesian models

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
   
    
