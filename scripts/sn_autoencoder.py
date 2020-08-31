#!/usr/bin/env python

import numpy as np
import argparse, os
import discrete1.setup as s
import discrete1.ae_prob as ae

parser = argparse.ArgumentParser(description='Enrichment')
parser.add_argument('-enrich',action='store',dest='en',nargs='+') #Enrichment looking into
parser.add_argument('-problem',action='store',dest='problem') # Problem set up
parser.add_argument('-model',action='store',dest='model')
parser.add_argument('-nodes',action='store',dest='nodes',nargs='+')
parser.add_argument('-auto',action='store',dest='autoencoder')
parser.add_argument('-gpu',action='store',dest='gpu')
parser.add_argument('-mult',action='store',dest='mult')
parser.add_argument('-source',action='store',dest='source')
usr_input = parser.parse_args()

# if usr_input.gpu is None:
    # print('No GPU')
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# else:
    # print('Using GPU')

enrich = [float(jj) for jj in usr_input.en]
labels = [str(jj).split('.')[1] for jj in enrich]

if usr_input.autoencoder is None:
    print('Encode-Decode Model')
    label2 = 'eNDe_'
else:
    print('Autoencoder Model')
    label2 = 'ae_'

sprob = '{}_full'.format(usr_input.problem)


if usr_input.mult == 'both':
    print('Using Fission and Scatter AEs')
    label3 = 'mult_'
elif usr_input.mult is None:
    label3 = 'phi_'
else:
    label3 = usr_input.mult + '_'

if usr_input.source is None:
    label4 = ''
else:
    label4 = '_source'

if 'dummy' not in usr_input.model:
    nodes = '-'.join([str(int(ii)) for ii in usr_input.nodes])
    ae_model = 'autoencoder/{}{}/model{}'.format(usr_input.model,label4,nodes)
    print('Autoencoder Model {}'.format(ae_model))
    folder = 'mydata/ae{}_{}/{}'.format(label4,usr_input.problem,usr_input.model)
    file = '_{}{}n{}_e'.format(label2,label3,nodes)
else:
    ae_model = usr_input.model
    print('Dummy Model {}'.format(usr_input.model))
    folder = 'mydata/{}{}_true_1d'.format(label2,label4)
    file = '_{}{}_'.format(label3,ae_model)


for ii in range(len(enrich)):
    if usr_input.source is None:
        enrichment,splits = s.problem1.boundaries(enrich[ii],problem=usr_input.problem)
        if usr_input.autoencoder is None:
            problem = ae.eigen_eNDe(*s.problem1.variables(enrich[ii],problem=usr_input.problem))
            phi,keff = problem.transport(ae_model,problem=usr_input.problem)
        else:
            problem = ae.eigen_auto(*s.problem1.variables(enrich[ii],ptype=usr_input.problem,symm=True))
            phi,keff = problem.transport(ae_model,problem=usr_input.problem,LOUD=True,multAE=usr_input.mult)
        np.save('{}/phi{}{:<02}'.format(folder,file,labels[ii]),phi)
        np.save('{}/keff{}{:<02}'.format(folder,file,labels[ii]),keff)
    else:
        if usr_input.autoencoder is None:
            problem = ae.source_eNDe(*s.problem1.variables(enrich[ii],problem=usr_input.problem))
        else:
            problem = ae.source_auto(*s.problem1.variables(enrich[ii],problem=usr_input.problem))
        phi = problem.transport(ae_model,problem=usr_input.problem+'_source')
        np.save('{}/phi{}{:<02}'.format(folder,file,labels[ii]),phi)
        


    