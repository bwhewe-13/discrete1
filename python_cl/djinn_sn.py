#!/usr/bin/env python

import numpy as np
import discrete1.slab as s
import discrete1.initialize as ex
from discrete1.util import nnets
import argparse

parser = argparse.ArgumentParser(description='Enrichment')
# parser.add_argument('-e',action='store',dest='en',type=float)
parser.add_argument('-e',action='store',dest='en',nargs='+')
parser.add_argument('-d',action='store',dest='dist',nargs='+')
parser.add_argument('-xs',action='store',dest='xs')
usr_input = parser.parse_args()

enrich = [float(jj) for jj in usr_input.en]
if usr_input.dist is None:
    print('Using Default Dimensions')
file1 = usr_input.xs
# label = str(enrich).split('.')[1]
labels = [str(jj).split('.')[1] for jj in enrich]

if file1 == 'both':
    djinn_model = []    
    nums = nnets.djinn_metric('{}_enrich_1d_djinn_model/error/model*'.format('scatter'),clean=True)
    djinn_model.append('{}_enrich_1d_djinn_model/model_{}'.format('scatter',nums))
    nums = nnets.djinn_metric('{}_enrich_1d_djinn_model/error/model*'.format('fission'),clean=True)
    djinn_model.append('{}_enrich_1d_djinn_model/model_{}'.format('fission',nums))
else:
    nums = nnets.djinn_metric('{}_enrich_1d_djinn_model/error/model*'.format(file1),clean=True)
    djinn_model = '{}_enrich_1d_djinn_model/model_{}'.format(file1,nums)

for ii in range(len(enrich)):
    enrichment,splits = ex.eigen_djinn.variables(enrich[ii],distance=usr_input.dist,symm=True)
    problem = s.eigen_djinn_symm(*ex.eigen_djinn.variables(enrich[ii],distance=usr_input.dist,symm=True),dtype=file1,enrich=enrichment,splits=splits)
    phi,keff = problem.transport(djinn_model,LOUD=True)
    np.save('mydata/djinn_{}_1d/phi_enrich_{:<02}'.format(file1,labels[ii]),phi)
    np.save('mydata/djinn_{}_1d/keff_enrich_{:<02}'.format(file1,labels[ii]),keff)   
