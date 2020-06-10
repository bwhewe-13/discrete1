#!/usr/bin/env python

import numpy as np
from djinn import djinn
import sklearn
import argparse
from whewell import auto_mail

parser = argparse.ArgumentParser(description='Which Search')
parser.add_argument('-e',action='store',dest='en')
usr_input = parser.parse_args()

if usr_input.en == 's':
    print('Scattering')
    file1 = 'scatter'
elif usr_input.en == 'f':
    print('Fission')
    file1 = 'fission'


mymat = np.load('mydata/djinn_true_1d/true_track_{}_combined.npy'.format(file1))
X = mymat[:,:2,:].copy() #check this
X = np.hstack((X[:,0][:,:1],X[:,1]))
Y = mymat[:,2,:].copy()

num_trees = [1,3,5] 
num_depth = [2,4,6] 

split = 0.2
spatial = np.arange(len(mymat))
np.random.shuffle(spatial)
np.save('mydata/djinn_{}_1d/spatialShuffleMat'.format(file1),spatial)
index = int(len(spatial)*split)

x_train = X[spatial[index:]]; y_train = Y[spatial[index:]]
x_test = X[spatial[:index]]; y_test = Y[spatial[:index]]


for jj in num_trees:
    for ii in num_depth:
        print("ntrees:",jj,"maxdepth:",ii)
        print('='*25)
        z_jj = str(jj).zfill(3); z_ii = str(ii).zfill(3)
        modelname="{}_enrich_1d_djinn_model/model_{}{}".format(file1,z_jj,z_ii)    # name the model
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
                      display_step=1, save_files=True, file_name=modelname, 
                      save_model=True,model_name=modelname)

        m=model.predict(x_test)

        # evaluate results
        with open('{}_enrich_1d_djinn_model/error/model_ntrees{}_maxdepth{}.txt'.format(file1,z_jj,z_ii),'a') as f:
            for kk in range(y_test.shape[0]):
                mse=sklearn.metrics.mean_squared_error(y_test[kk],m[kk])
                mabs=sklearn.metrics.mean_absolute_error(y_test[kk],m[kk])
                exvar=sklearn.metrics.explained_variance_score(y_test[kk],m[kk])   
                f.write('MSE '+str(mse)+' M Abs Err '+str(mabs)+' Expl. Var. '+str(exvar)+'\n')
        # close model 
        model.close_model()

auto_mail.send("DJINN Finished - {}".format(file1))       
    