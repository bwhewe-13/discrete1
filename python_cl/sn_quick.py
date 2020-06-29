#!/usr/bin/env python

import numpy as np
import discrete1.rogue as r
from discrete1.util import nnets
import argparse

parser = argparse.ArgumentParser(description='Enrichment')
parser.add_argument('-enrich',action='store',dest='en',nargs='+') #Enrichment looking into
parser.add_argument('-distance',action='store',dest='dist',nargs='+') # Changes the dimensions of the original problem
parser.add_argument('-xs',action='store',dest='xs') # Which matrix multiply DJINN estimates, 'fission','scatter', or 'both'
parser.add_argument('-label',action='store',dest='label') # If the DJINN model is with labeled data
parser.add_argument('-track',action='store',dest='track') # Tracking the fission and scattering data for DJINN iterations
parser.add_argument('-norm',action='store',dest='normed')
usr_input = parser.parse_args()

enrich = [float(jj) for jj in usr_input.en]
if usr_input.dist is None:
    print('Using Default Dimensions')
file1 = usr_input.xs
# label = str(enrich).split('.')[1]
labels = [str(jj).split('.')[1] for jj in enrich]

if usr_input.normed == 'norm':
    file2 = 'normed/'
elif usr_input.normed == 'small':
    file2 = 'small/'
else:
    file2 = ''
    
if usr_input.label is None:
    file3 = '_reg/'
else:
    file3 = '_label/'

# if usr_input.label is None:
    # file2 = ''
    # file3 = '_reg' # Does not include labeled data
# else:
    # file2 = '_enrich'
    # file3 = '_enrich'

if file1 == 'both':
    djinn_model = []    
    nums = nnets.djinn_metric('{}_1d/{}djinn{}/error/model*'.format('scatter',file2,file3),clean=True)
    djinn_model.append('{}_1d/{}djinn{}/model_{}'.format('scatter',file2,file3,nums))
    nums = nnets.djinn_metric('{}_1d/{}djinn{}/error/model*'.format('fission',file2,file3),clean=True)
    djinn_model.append('{}_1d/{}djinn{}/model_{}'.format('fission',file2,file3,nums))
else:
    nums = nnets.djinn_metric('{}_1d/{}djinn{}/error/model*'.format(file1,file2,file3),clean=True)
    djinn_model = '{}_1d/{}djinn{}/model_{}'.format(file1,file2,file3,nums)
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
    np.save('mydata/djinn_{}_1d/phi_{}{}_{:<02}'.format(file1,file2[:len(file2)-1],file3[:len(file3)-1],labels[ii]),phi)
    np.save('mydata/djinn_{}_1d/keff_{}{}_{:<02}'.format(file1,file2[:len(file2)-1],file3[:len(file3)-1],labels[ii]),keff)   