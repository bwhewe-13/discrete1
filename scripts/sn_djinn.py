#!/usr/bin/env python

import numpy as np
from discrete1.dj_prob import eigen_djinn,source_djinn
from discrete1.util import nnets
import discrete1.setup as s
import argparse, os

parser = argparse.ArgumentParser(description='Enrichment')

parser.add_argument('-problem',action='store',dest='problem') 
parser.add_argument('-fmodel',action='store',dest='fmodel',nargs='+')
parser.add_argument('-smodel',action='store',dest='smodel',nargs='+')
parser.add_argument('-source',action='store',dest='source')
parser.add_argument('-enrich',action='store',dest='en',nargs='+') 
parser.add_argument('-track',action='store',dest='track') 
usr_input = parser.parse_args()

# Enrichment and associated labels
enrich = [float(jj) for jj in usr_input.en]
labels = [str(jj).split('.')[1] for jj in enrich]

source_add = ''
if usr_input.source:
    source_add = 'src_'

if usr_input.fmodel:
    fmodel_file = '{}fission_1d/'.format(source_add)+usr_input.fmodel[0]+'/djinn_'+usr_input.fmodel[1]
if usr_input.smodel:
    smodel_file = '{}scatter_1d/'.format(source_add)+usr_input.smodel[0]+'/djinn_'+usr_input.smodel[1]

labeled = False; djinn_model = []
# if using both the systems
if usr_input.fmodel and usr_input.smodel:
    if usr_input.fmodel[1] == 'label':
        labeled = True
    nums = nnets.djinn_metric('{}/error/model*'.format(smodel_file),clean=True)
    djinn_model.append('{}/model_{}'.format(smodel_file,nums))
    nums = nnets.djinn_metric('{}/error/model*'.format(fmodel_file),clean=True) 
    djinn_model.append('{}/model_{}'.format(fmodel_file,nums))
    save_folder = 'mydata/djinn_{}/both'.format(usr_input.smodel[0]); save_file = usr_input.smodel[1]
# if using only scatter
elif usr_input.smodel:
    if usr_input.smodel[1] == 'label':
        labeled = True
    nums = nnets.djinn_metric('{}/error/model*'.format(smodel_file),clean=True)
    djinn_model = '{}/model_{}'.format(smodel_file,nums)
    save_folder = 'mydata/djinn_{}/scatter'.format(usr_input.smodel[0]); save_file = usr_input.smodel[1]
# if using only fission
elif usr_input.fmodel:
    if usr_input.fmodel[1] == 'label':
        labeled = True
    nums = nnets.djinn_metric('{}/error/model*'.format(fmodel_file),clean=True)
    djinn_model = '{}/model_{}'.format(fmodel_file,nums)
    save_folder = 'mydata/djinn_{}/fission'.format(usr_input.fmodel[0]); save_file = usr_input.fmodel[1]

print('DJINN Model',djinn_model)
multDJ = save_folder.split('/')[2]

for ii in range(len(enrich)):
    enrichment,splits = s.problem.boundaries(enrich[ii],problem=usr_input.problem)
    print(splits)
    if usr_input.source is None:
        problem = eigen_djinn(*s.problem.variables(enrich[ii],problem=usr_input.problem),enrich=enrichment,splits=splits,track=usr_input.track,label=labeled)
        phi,keff = problem.transport(djinn_model,problem=usr_input.problem,multDJ=multDJ)
        np.save('{}_keff{}_{:<02}'.format(save_folder,save_file,labels[ii]),keff)
    else:
        print('Source Problem')
        problem = source_djinn(*s.problem.variables(enrich[ii],problem=usr_input.problem),enrich=enrichment,splits=splits,label=labeled)
        phi = problem.transport(djinn_model,problem=usr_input.problem,multDJ=multDJ)    
    np.save('{}_phi{}_{:<02}'.format(save_folder,save_file,labels[ii]),phi)
    
