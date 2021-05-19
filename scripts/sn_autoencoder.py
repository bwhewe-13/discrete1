#!/usr/bin/env python

import numpy as np
import argparse, os
from discrete1.critical import Critical

parser = argparse.ArgumentParser(description='Enrichment')
parser.add_argument('-enrich',action='store',dest='en',nargs='+') # Enrichment looking into
parser.add_argument('-model',action='store',dest='model') 	  # Either "phi" or "smult" 
parser.add_argument('-iteration',action='store',dest='iteration') # count
usr_input = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

enrich = [float(jj) for jj in usr_input.en]
labels = [str(jj).split('.')[1] for jj in enrich]

if usr_input.model == 'fmult':
    atype = 'fission'
else:
    atype = 'phi'

model_name = 'autoencoder/fission_pluto/iteration{}/model_300-150_{}'.format(usr_input.iteration,usr_input.model)
folder = 'mydata/ae_pluto_fiss'
file = '_{}_iter{}_e'.format(usr_input.model,usr_input.iteration)

print(folder)
print(file)

for ii in range(len(enrich)):
    phi,keff = Critical.run_auto('pu',enrich[ii],model_name,atype,focus='fuel',transform='minmax')
    np.save('{}/phi{}{:<02}'.format(folder,file,labels[ii]),phi)
    np.save('{}/keff{}{:<02}'.format(folder,file,labels[ii]),keff)

    
# multAE = 'smult'
# # multAE = usr_input.model
# model_name = 'autoencoder/{}_011/model_{}'.format(usr_input.problem,nodes)
# folder = 'mydata/ae_{}_012'.format(usr_input.problem) # ,usr_input.model
# file = '_{}_n{}_e'.format(multAE,nodes)

# setup = problem1
# if usr_input.problem == 'pluto':
#     setup = problem2

# for ii in range(len(enrich)):
#     enrichment,splits = setup.boundaries(enrich[ii],problem=usr_input.problem)
#     problem = eigen_auto(*setup.variables(enrich[ii],problem=usr_input.problem))
#     phi,keff = problem.transport(model_name,number=usr_input.number,problem=usr_input.problem)
#     np.save('{}/phi{}{:<02}_{}'.format(folder,file,labels[ii],usr_input.number),phi)
#     np.save('{}/keff{}{:<02}_{}'.format(folder,file,labels[ii],usr_input.number),keff)
