#!/usr/bin/env python

import numpy as np
import discrete1.hoth as h
import argparse
import os

parser = argparse.ArgumentParser(description='Enrichment')
parser.add_argument('-enrich',action='store',dest='en',nargs='+') #Enrichment looking into
parser.add_argument('-problem',action='store',dest='problem') # Problem set up
parser.add_argument('-model',action='store',dest='model')
parser.add_argument('-nodes',action='store',dest='nodes',nargs='+')
parser.add_argument('-gpu',action='store',dest='gpu')
usr_input = parser.parse_args()

if usr_input.gpu is None:
    print('No GPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    print('Using GPU')

enrich = [float(jj) for jj in usr_input.en]
labels = [str(jj).split('.')[1] for jj in enrich]

sprob = '{}_full'.format(usr_input.problem)
nodes = '-'.join([int(ii) for ii in usr_input.nodes])
ae_model = 'autoencoder/{}/model{}'.format(usr_input.model,nodes)

for ii in range(len(enrich)):
    enrichment,splits = h.problem.boundaries(enrich[ii],ptype1=usr_input.problem,ptype2=sprob,symm=True)
    problem = h.eigen_auto(*r.problem.variables(enrich[ii],ptype=usr_input.problem,symm=True),track=usr_input.track)
    phi,keff = problem.transport(ae_model,problem=usr_input.problem,LOUD=True)
    np.save('mydata/ae_{}/phi_{}{}_{:<02}'.format(usr_input.problem,usr_input.model,nodes,labels[ii]),phi)
    np.save('mydata/ae_{}/keff_{}{}_{:<02}'.format(usr_input.problem,usr_input.model,nodes,labels[ii]),keff)


    