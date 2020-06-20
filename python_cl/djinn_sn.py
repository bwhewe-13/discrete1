#!/usr/bin/env python

import numpy as np
import discrete1.slab as s
import discrete1.initialize as ex
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
    nums = nnets.djinn_metric('{}{}_1d_djinn_model/error/model*'.format('scatter',file2),clean=True)
    djinn_model.append('{}{}_1d_djinn_model/model_{}'.format('scatter',file2,nums))
    nums = nnets.djinn_metric('{}{}_1d_djinn_model/error/model*'.format('fission',file2),clean=True)
    djinn_model.append('{}{}_1d_djinn_model/model_{}'.format('fission',file2,nums))
else:
    nums = nnets.djinn_metric('{}{}_1d_djinn_model/error/model*'.format(file1,file2),clean=True)
    djinn_model = '{}{}_1d_djinn_model/model_{}'.format(file1,file2,nums)
print('DJINN Model',djinn_model)
for ii in range(len(enrich)):
    enrichment,splits = ex.eigen_djinn.boundaries(enrich[ii],distance=usr_input.dist,symm=True)
    problem = s.eigen_djinn_symm(*ex.eigen_djinn.variables(enrich[ii],distance=usr_input.dist,symm=True),dtype=file1,enrich=enrichment,splits=splits,track=usr_input.track,label=usr_input.label)
    if usr_input.track is None:
        phi,keff = problem.transport(djinn_model,LOUD=True)
    else:
        phi,keff,fission,scatter = problem.transport(djinn_model,LOUD=True)
        np.save('mydata/djinn_{}_1d/{}_{:<02}'.format(file1,labels[ii]),fission)
        np.save('mydata/djinn_{}_1d/{}_{:<02}'.format(file1,labels[ii]),scatter)
    np.save('mydata/djinn_{}_1d/phi{}_{:<02}'.format(file1,file3,labels[ii]),phi)
    np.save('mydata/djinn_{}_1d/keff{}_{:<02}'.format(file1,file3,labels[ii]),keff)   


# %% testing 
# import numpy as np
# import discrete1.slab as s
# import discrete1.initialize as ex
# from discrete1.util import nnets
# enrich = [0.05]
# file1 = 'both'
# # file2 = '_enrich'
# file2 = ''
# label=None
# if file1 == 'both':
#     djinn_model = []    
#     nums = nnets.djinn_metric('{}{}_1d_djinn_model/error/model*'.format('scatter',file2),clean=True)
#     djinn_model.append('{}{}_1d_djinn_model/model_{}'.format('scatter',file2,nums))
#     nums = nnets.djinn_metric('{}{}_1d_djinn_model/error/model*'.format('fission',file2),clean=True)
#     djinn_model.append('{}{}_1d_djinn_model/model_{}'.format('fission',file2,nums))
# else:
#     nums = nnets.djinn_metric('{}{}_1d_djinn_model/error/model*'.format(file1,file2),clean=True)
#     djinn_model = '{}{}_1d_djinn_model/model_{}'.format(file1,file2,nums)

# for ii in range(len(enrich)):
#     enrichment,splits = ex.eigen_djinn.boundaries(enrich[ii],symm=True)
#     problem = s.eigen_djinn_symm(*ex.eigen_djinn.variables(enrich[ii],symm=True),dtype=file1,enrich=enrichment,splits=splits,track='both',label=label)
#     phi,keff,fis,sca = problem.transport(djinn_model,LOUD=True)
    
