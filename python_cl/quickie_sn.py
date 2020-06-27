#!/usr/bin/env python

import numpy as np
import discrete1.rogue as r
from discrete1.util import nnets
import argparse

parser = argparse.ArgumentParser(description='Enrichment')
# parser.add_argument('-e',action='store',dest='en',type=float)
parser.add_argument('-e',action='store',dest='en',nargs='+') #Enrichment looking into
parser.add_argument('-d',action='store',dest='dist',nargs='+') # Changes the dimensions of the original problem
parser.add_argument('-xs',action='store',dest='xs') # Which matrix multiply DJINN estimates, 'fission','scatter', or 'both'
parser.add_argument('-l',action='store',dest='label') # If the DJINN model is with labeled data
parser.add_argument('-track',action='store',dest='track') # Tracking the fission and scattering data for DJINN iterations
usr_input = parser.parse_args()

enrich = [float(jj) for jj in usr_input.en]
if usr_input.dist is None:
    print('Using Default Dimensions')
file1 = usr_input.xs
# label = str(enrich).split('.')[1]
labels = [str(jj).split('.')[1] for jj in enrich]

if usr_input.label is None:
    file2 = ''
    file3 = '_reg' # Does not include labeled data
else:
    file2 = '_enrich'
    file3 = '_enrich'

if file1 == 'both':
    djinn_model = []    
    nums = nnets.djinn_metric('{}2{}_1d_djinn_model/error/model*'.format('scatter',file2),clean=True)
    djinn_model.append('{}2{}_1d_djinn_model/model_{}'.format('scatter',file2,nums))
    nums = nnets.djinn_metric('{}2{}_1d_djinn_model/error/model*'.format('fission',file2),clean=True)
    djinn_model.append('{}2{}_1d_djinn_model/model_{}'.format('fission',file2,nums))
else:
    nums = nnets.djinn_metric('{}2{}_1d_djinn_model/error/model*'.format(file1,file2),clean=True)
    djinn_model = '{}2{}_1d_djinn_model/model_{}'.format(file1,file2,nums)
print('DJINN Model',djinn_model)
for ii in range(len(enrich)):
    enrichment,splits = r.eigen_djinn.boundaries(enrich[ii],distance=usr_input.dist,symm=True)
    problem = r.eigen_djinn_symm(*r.eigen_djinn.variables(enrich[ii],distance=usr_input.dist,symm=True),dtype=file1,enrich=enrichment,splits=splits,track=usr_input.track,label=usr_input.label)
    phi,keff = problem.transport(djinn_model,LOUD=True)
    # if usr_input.track is None:
    #     phi,keff = problem.transport(djinn_model,LOUD=True)
    # else:
    #     phi,keff,fission,scatter = problem.transport(djinn_model,LOUD=True)
    #     np.save('mydata/djinn_{}_1d/{}_{:<02}'.format(file1,labels[ii]),fission)
    #     np.save('mydata/djinn_{}_1d/{}_{:<02}'.format(file1,labels[ii]),scatter)
    np.save('mydata/djinn_{}_1d/phi2{}_{:<02}'.format(file1,file3,labels[ii]),phi)
    np.save('mydata/djinn_{}_1d/keff2{}_{:<02}'.format(file1,file3,labels[ii]),keff)   