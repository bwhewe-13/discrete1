#!/usr/bin/env python

import numpy as np
from djinn import djinn
import sklearn
import argparse
import glob,os,json

parser = argparse.ArgumentParser(description='Which Search')
parser.add_argument('-xs',action='store',dest='xs')
parser.add_argument('-label',action='store',dest='label')
parser.add_argument('-model',action='store',dest='model')
parser.add_argument('-data',action='store',dest='data')
parser.add_argument('-gpu',action='store',dest='gpu')
parser.add_argument('-start',action='store',dest='start')
parser.add_argument('-stop',action='store',dest='stop')
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
    
if usr_input.model is None:
    file2 = ''
else:
    file2 = usr_input.model+'/'

start = int(usr_input.start)
stop = int(usr_input.stop)

address = np.sort(glob.glob('mydata/scatter_parts/mp*'))[start:stop]
print(address[0],address[-1])

mymat = np.load(address[0])

if usr_input.label is None:
    # X = mymat[0,:,1:].copy()
    # Y = mymat[1,:,1:].copy()
    file3 = '_reg/'
else:
    print('Labeled Data')
    # X = mymat[0].copy()
    # Y = mymat[1,:,1:].copy() # Remove labels
    file3 = '_label/'

# num_trees = [1,3,5]
num_trees = [3,5] 
num_depth = [2,6] 
# num_depth = [2,4,6] 


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

        # find optimal settings
        with open('{}_1d/{}djinn{}model_{}{}_parameters.json'.format(file1,file2,file3,z_jj,z_ii),'r') as fp:
            optimal = json.load(fp)

        batchsize=optimal['batch_size']
        learnrate=optimal['learn_rate']
        epochs=np.min((300,optimal['epochs']))
    
        for bb in range(len(address)):
            model2 = djinn.load(model_name=modelname,model_path=model_path)
            mymat = np.load(address[bb])
            if usr_input.label is None:
                X = mymat[0,:,1:].copy()
                Y = mymat[1,:,1:].copy()
            else:
                X = mymat[0].copy()
                Y = mymat[1,:,1:].copy() # Remove labels
            print(address[bb])
            x_train = X[:600000]; y_train = Y[:600000]
            x_test = X[600000:]; y_test = Y[600000:]
            model2.continue_training(x_train,y_train,epochs,learnrate,batchsize)
            m2 = model2.predict(x_test)
            with open('{}error/model_ntrees{}_maxdepth{}.txt'.format(model_path,z_jj,z_ii),'a') as f:
                for kk in range(y_test.shape[0]):
                    mse=sklearn.metrics.mean_squared_error(y_test[kk],m2[kk])
                    mabs=sklearn.metrics.mean_absolute_error(y_test[kk],m2[kk])
                    exvar=sklearn.metrics.explained_variance_score(y_test[kk],m2[kk])   
                    f.write('MSE '+str(mse)+' M Abs Err '+str(mabs)+' Expl. Var. '+str(exvar)+'\n')
            # close model 
            model2.close_model()
       
