#!/usr/bin/env python

import numpy as np
import discrete1.rogue as r
# import discrete1.plastic as r
from discrete1.util import nnets
import argparse
import os

parser = argparse.ArgumentParser(description='Enrichment')
parser.add_argument('-enrich',action='store',dest='en',nargs='+') #Enrichment looking into
parser.add_argument('-distance',action='store',dest='dist',nargs='+') # Changes the dimensions of the original problem
parser.add_argument('-xs',action='store',dest='xs') # Which matrix multiply DJINN estimates, 'fission','scatter', or 'both'
parser.add_argument('-label',action='store',dest='label') # If the DJINN model is with labeled data
parser.add_argument('-track',action='store',dest='track') # Tracking the fission and scattering data for DJINN iterations
parser.add_argument('-problem',action='store',dest='problem') # Problem set up
parser.add_argument('-fmodel',action='store',dest='fmodel')
parser.add_argument('-smodel',action='store',dest='smodel')
parser.add_argument('-gpu',action='store',dest='gpu')
# parser.add_argument('-scatter',action='store',dest='scatter')
usr_input = parser.parse_args()

if usr_input.gpu is None:
    print('No GPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    print('Using GPU')

enrich = [float(jj) for jj in usr_input.en]
labels = [str(jj).split('.')[1] for jj in enrich]

if usr_input.dist is None:
    print('Using Default Dimensions')

file1 = usr_input.xs
model = usr_input.smodel

process = None

try:
    ffile2 = usr_input.fmodel+'/'
except TypeError:
    ffile2 = ''
try:
    sfile2 = usr_input.smodel+'/'
except TypeError:
    sfile2 = ''

if file1 == "scatter":
    file2 = sfile2
elif file1 == 'fission':
    file2 = ffile2
    model = usr_input.fmodel

if 'norm' in model:
    process = 'norm'
    print('Norm Process')

if usr_input.label is None:
    file3 = '_reg'
else:
    file3 = '_label'

if file1 == 'both':
    djinn_model = []    
    nums = nnets.djinn_metric('{}_1d/{}djinn{}/error/model*'.format('scatter',sfile2,file3),clean=True)
    djinn_model.append('{}_1d/{}djinn{}/model_{}'.format('scatter',sfile2,file3,nums))
    nums = nnets.djinn_metric('{}_1d/{}djinn{}/error/model*'.format('fission',ffile2,file3),clean=True) #space should be file2
    djinn_model.append('{}_1d/{}djinn{}/model_{}'.format('fission',ffile2,file3,nums))
else:
    nums = nnets.djinn_metric('{}_1d/{}djinn{}/error/model*'.format(file1,file2,file3),clean=True)
    # if usr_input.scatter is not None:
    #     nums = str(usr_input.scatter)
    djinn_model = '{}_1d/{}djinn{}/model_{}'.format(file1,file2,file3,nums)

print('DJINN Model',djinn_model)

sprob = '{}_full'.format(usr_input.problem)

for ii in range(len(enrich)):
    enrichment,splits = r.problem.boundaries(enrich[ii],distance=usr_input.dist,ptype1=usr_input.problem,ptype2=sprob,symm=True)
    print(splits)
    problem = r.eigen_djinn_symm(*r.problem.variables(enrich[ii],ptype=usr_input.problem,distance=usr_input.dist,symm=True),dtype=file1,enrich=enrichment,splits=splits,track=usr_input.track,label=usr_input.label)
    # phi,keff = problem.transport(djinn_model,LOUD=True,MAX_ITS=1)
    # if usr_input.track is None:
        # phi,keff = problem.transport(djinn_model,process=process,LOUD=True)
    # else:
        # phi,keff,track_fission,track_scatter = problem.transport(djinn_model,process=process,LOUD=True)
        # np.save('mydata/djinn_{}_1d/{}_{}{}_{:<02}'.format(file1,file1,usr_input.normed,file3,labels[ii]),track_fission)
        # np.save('mydata/djinn_{}_1d/{}_{}{}_{:<02}'.format(file1,file1,usr_input.normed,file3,labels[ii]),track_scatter)
    phi,keff = problem.transport(djinn_model,process=process,ptype=usr_input.problem,LOUD=True)
    np.save('mydata/{}/{}_phi{}_{:<02}'.format(model,file1,file3,labels[ii]),phi)
    np.save('mydata/{}/{}_keff{}_{:<02}'.format(model,file1,file3,labels[ii]),keff)


    
