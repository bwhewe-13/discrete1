#!/usr/bin/env python

import numpy as np
import argparse, os
from discrete1.setup import problem2
from discrete1.ae_prob import eigen_auto_djinn

parser = argparse.ArgumentParser(description='Enrichment')
parser.add_argument('-enrich',action='store',dest='en',nargs='+') #Enrichment looking into
parser.add_argument('-problem',action='store',dest='problem') # Problem set up
# parser.add_argument('-model',action='store',dest='model')
parser.add_argument('-nodes',action='store',dest='nodes',nargs='+')
parser.add_argument('-part',action='store',dest='part')
parser.add_argument('-num',action='store',dest='num')
# parser.add_argument('-mult',action='store',dest='mult')
usr_input = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

enrich = [float(jj) for jj in usr_input.en]
labels = [str(jj).split('.')[1] for jj in enrich]

nodes = '-'.join([str(jj) for jj in usr_input.nodes])

djinn_model = 'scatter_1d/pluto_{}/djinn_reg/model_{}'.format(usr_input.part,usr_input.num)

multAE = 'scatter'
# multAE = usr_input.model
ae_model = 'autoencoder/{}_{}/model_{}'.format(usr_input.problem,usr_input.part,nodes)
folder = 'mydata/combo_{}'.format(usr_input.problem) # ,usr_input.model
file = '_part{}_djinn{}_e'.format(usr_input.part,usr_input.num)


for ii in range(len(enrich)):
    distance = [5,1.5,3.5]; dim = 618
    enrichment,splits = problem2.boundaries(enrich[ii],problem=usr_input.problem)
    print(splits)
    problem = eigen_auto_djinn(*problem2.variables(enrich[ii],dim=dim,distance=distance),enrich=enrichment,splits=splits)
    # problem = eigen_auto_djinn(*problem2.variables(enrich[ii],problem=usr_input.problem))
    phi,keff = problem.transport(ae_model,djinn_model,problem=usr_input.problem,multAE=multAE)
    np.save('{}/phi{}{:<02}'.format(folder,file,labels[ii]),phi)
    np.save('{}/keff{}{:<02}'.format(folder,file,labels[ii]),keff)

    
    