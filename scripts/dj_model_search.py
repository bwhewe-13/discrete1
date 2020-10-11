#!/usr/bin/env python

import numpy as np
from djinn import djinn
import sklearn
from sklearn import model_selection
import argparse
import glob, os

parser = argparse.ArgumentParser(description='Which Search')
parser.add_argument('-xs',action='store',dest='xs')
parser.add_argument('-label',action='store',dest='label')
parser.add_argument('-model',action='store',dest='model')
parser.add_argument('-data',action='store',dest='data')
parser.add_argument('-gpu',action='store',dest='gpu')
# parser.add_argument('-source',action='store',dest='source')
# data can be orig, red, inter
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
    # Normalize
    if 'norm' in usr_input.model:
        print('Normed, Non Labeled Data')
        X /= np.linalg.norm(X,axis=1)[:,None]
        # file2 = 'normed/'
    else:
        print('Non Labeled Data')
        # file2 = ''
    Y = mymat[1,:,1:].copy()
    file3 = '_reg/'
else:
    print('Labeled Data')
    if 'norm' in usr_input.model:
        print('Normed, Labeled Data')
        tX = mymat[0,:,1:].copy()
        tX /= np.linalg.norm(tX,axis=1)[:,None]
        X = np.hstack((mymat[0,:,0][:,None],tX))
        # file2 = 'normed/'
    else:
        print('Labeled Data')
        X = mymat[0].copy()
        # file2 = ''
    Y = mymat[1,:,1:].copy() # Remove labels
    file3 = '_label/'

# num_trees = [1,3,5]
num_trees = [3,5] 
num_depth = [2,6] 
# num_depth = [2,4,6] 

split = 0.2
try:
    spatial = np.load('{}_1d/{}spatialShuffleMat.npy'.format(file1,file2))
except FileNotFoundError:
    spatial = np.arange(len(X))
    np.random.shuffle(spatial)
    np.save('{}_1d/{}spatialShuffleMat'.format(file1,file2),spatial)

index = int(len(spatial)*split)
x_train = X[spatial[index:]]; y_train = Y[spatial[index:]]
x_test = X[spatial[:index]]; y_test = Y[spatial[:index]]
# print(sklearn.__version__)
# x_train,x_test,y_train,y_test = model_selection.train_test_split(X,Y,test_size=0.2,random_state=47)

for jj in num_trees:
    for ii in num_depth:
        print("ntrees:",jj,"maxdepth:",ii)
        print('='*25)
        z_jj = str(jj).zfill(3); z_ii = str(ii).zfill(3)
        modelname="model_{}{}".format(z_jj,z_ii) 
        model_path = '{}_1d/{}djinn{}'.format(file1,file2,file3)
        ntrees=jj                        # number of trees = number of neural nets in ensemble
        maxdepth=ii                      # max depth of tree -- optimize this for each data set
        dropout_keep=1.0                # dropout typically set to 1 for non-Bayesian models

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
   
    
