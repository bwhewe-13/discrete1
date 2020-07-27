#!/usr/bin/env python

import numpy as np
import discrete1.rogue as r
# import discrete1.plastic as r
from discrete1.util import nnets
import argparse

parser = argparse.ArgumentParser(description='Enrichment')
parser.add_argument('-enrich',action='store',dest='en',nargs='+') #Enrichment looking into
parser.add_argument('-distance',action='store',dest='dist',nargs='+') # Changes the dimensions of the original problem
parser.add_argument('-xs',action='store',dest='xs') # Which matrix multiply DJINN estimates, 'fission','scatter', or 'both'
parser.add_argument('-label',action='store',dest='label') # If the DJINN model is with labeled data
parser.add_argument('-track',action='store',dest='track') # Tracking the fission and scattering data for DJINN iterations
parser.add_argument('-problem',action='store',dest='problem') # Problem set up
parser.add_argument('-model',action='store',dest='model')
usr_input = parser.parse_args()

enrich = [float(jj) for jj in usr_input.en]
if usr_input.dist is None:
    print('Using Default Dimensions')
file1 = usr_input.xs
# label = str(enrich).split('.')[1]
labels = [str(jj).split('.')[1] for jj in enrich]

process = None
if usr_input.model is None:
    file2 = ''
else:
    if 'norm' in usr_input.model:
        process = 'norm'
        print('Norm Process')
    file2 = usr_input.model+'/'
        
if usr_input.label is None:
    file3 = '_reg'
else:
    file3 = '_label'

if file1 == 'both':
    djinn_model = []    
    nums = nnets.djinn_metric('{}_1d/{}djinn{}/error/model*'.format('scatter',file2,file3),clean=True)
    djinn_model.append('{}_1d/{}djinn{}/model_{}'.format('scatter',file2,file3,nums))
    nums = nnets.djinn_metric('{}_1d/{}djinn{}/error/model*'.format('fission',file2,file3),clean=True)
    djinn_model.append('{}_1d/{}djinn{}/model_{}'.format('fission',file2,file3,nums))
else:
    nums = nnets.djinn_metric('{}_1d/{}djinn{}/error/model*'.format(file1,file2,file3),clean=True)
    # nums = '003004'
    # print(nums)
    djinn_model = '{}_1d/{}djinn{}/model_{}'.format(file1,file2,file3,nums)
print('DJINN Model',djinn_model)

for ii in range(len(enrich)):
    enrichment,splits = r.problem.boundaries(enrich[ii],distance=usr_input.dist,ptype=usr_input.problem,symm=True)
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
    np.save('mydata/djinn_{}_1d/phi_{}{}_{:<02}'.format(file1,usr_input.model,file3,labels[ii]),phi)
    np.save('mydata/djinn_{}_1d/keff_{}{}_{:<02}'.format(file1,usr_input.model,file3,labels[ii]),keff)
    # np.save('mydata/djinn_{}_1d/phi_{}{}_{:<02}'.format(file1,usr_input.normed,file3,labels[ii]),phi)
    # np.save('mydata/djinn_{}_1d/keff_{}{}_{:<02}'.format(file1,usr_input.normed,file3,labels[ii]),keff)   
        

