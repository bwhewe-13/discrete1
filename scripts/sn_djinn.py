#!/usr/bin/env python

import numpy as np
from discrete1.dj_prob import eigen_djinn,source_djinn
from discrete1.util import nnets
from discrete1.setup import problem1,problem2
import argparse, os

parser = argparse.ArgumentParser(description='Enrichment')

parser.add_argument('-problem',action='store',dest='problem') 
parser.add_argument('-fmodel',action='store',dest='fmodel',nargs='+')
parser.add_argument('-smodel',action='store',dest='smodel',nargs='+')
parser.add_argument('-rmodel',action='store',dest='rmodel',nargs='+')
# parser.add_argument('-source',action='store',dest='source')
parser.add_argument('-enrich',action='store',dest='en',nargs='+') 
parser.add_argument('-track',action='store',dest='track') 
parser.add_argument('-num',action='store',dest='num')
parser.add_argument('-num2',action='store',dest='num2') # for reflective
usr_input = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Enrichment and associated labels
enrich = [float(jj) for jj in usr_input.en]
labels = [str(jj).split('.')[1] for jj in enrich]

# source_add = ''
# if usr_input.source:
#     source_add = 'src_'

# if usr_input.fmodel:
#     fmodel_file = '{}fission_1d/'.format(source_add)+usr_input.fmodel[0]+'/djinn_'+usr_input.fmodel[1]
# if usr_input.smodel:
#     smodel_file = '{}scatter_1d/'.format(source_add)+usr_input.smodel[0]+'/djinn_'+usr_input.smodel[1]

# labeled = False; djinn_model = []
# # if using both the systems
# if usr_input.fmodel and usr_input.smodel:
#     if usr_input.fmodel[1] == 'label':
#         labeled = True
#     # nums = '003002'
#     nums = nnets.djinn_metric('{}/error/model*'.format(smodel_file),clean=True)
#     djinn_model.append('{}/model_{}'.format(smodel_file,nums))
#     # nums = '003002'
#     nums = nnets.djinn_metric('{}/error/model*'.format(fmodel_file),clean=True) 
#     djinn_model.append('{}/model_{}'.format(fmodel_file,nums))
#     save_folder = 'mydata/djinn_{}{}/both'.format(source_add,usr_input.smodel[0]); save_file = usr_input.smodel[1]
# # if using only scatter
# elif usr_input.smodel:
#     if usr_input.smodel[1] == 'label':
#         labeled = True
#     # nums = nnets.djinn_metric('{}/error/model*'.format(smodel_file),clean=True)
#     nums = usr_input.num
#     djinn_model = '{}/model_{}'.format(smodel_file,nums)
#     save_folder = 'mydata/djinn_{}/scatter_{}'.format(usr_input.smodel[0],nums); save_file = usr_input.smodel[1]
# # if using only fission
# elif usr_input.fmodel:
#     if usr_input.fmodel[1] == 'label':
#         labeled = True
#     # nums = nnets.djinn_metric('{}/error/model*'.format(fmodel_file),clean=True)
#     nums = usr_input.num
#     djinn_model = '{}/model_{}'.format(fmodel_file,nums)
#     # save_folder = 'mydata/djinn_{}{}/fission'.format(source_add,usr_input.fmodel[0]); save_file = usr_input.fmodel[1]
#     save_folder = 'mydata/djinn_{}/fission_{}'.format(usr_input.fmodel[0],nums); save_file = usr_input.fmodel[1]

labeled = False
if usr_input.smodel[1] == 'label':
    labeled = True
nums = usr_input.num
nums2 = usr_input.num2

fuel_model = 'scatter_1d/cscat/djinn_'+usr_input.smodel[1]+'/model_{}'.format(nums)
refl_model = 'scatter_1d/hdpe/djinn_'+usr_input.rmodel[1]+'/model_{}'.format(nums2)
save_folder = 'mydata/djinn_cscat_hdpe/scatter_{}_{}'.format(nums,nums2)
save_file = 'double'
djinn_model = [fuel_model,refl_model]


print('DJINN Model',djinn_model)
# multDJ = save_folder.split('/')[2]
print(save_folder)
print(save_file)

multDJ = 'scatter'
print(multDJ)


for ii in range(len(enrich)):
    enrichment,splits = problem1.boundaries(enrich[ii],problem=usr_input.problem)
    print(splits)
    problem = eigen_djinn(*problem1.variables(enrich[ii],problem=usr_input.problem),enrich=enrichment,splits=splits,track=usr_input.track,label=labeled)
    phi,keff = problem.transport(djinn_model,problem=usr_input.problem,multDJ=multDJ,double=True)
    np.save('{}_keff_{}_{:<02}'.format(save_folder,save_file,labels[ii]),keff)
    np.save('{}_phi_{}_{:<02}'.format(save_folder,save_file,labels[ii]),phi)
        

# if usr_input.problem == 'pluto':
#     for ii in range(len(enrich)):
#         distance = [5,1.5,3.5]; dim = 618
#         enrichment,splits = problem2.boundaries(enrich[ii],problem=usr_input.problem)
#         print(splits)
#         problem = eigen_djinn(*problem2.variables(enrich[ii],dim=dim,distance=distance),enrich=enrichment,track=usr_input.track,splits=splits,label=labeled)
#         phi,keff = problem.transport(djinn_model,problem=usr_input.problem,multDJ=multDJ)
#         np.save('{}_phi_{}_{:<02}'.format(save_folder,save_file,labels[ii]),phi)
#         np.save('{}_keff_{}_{:<02}'.format(save_folder,save_file,labels[ii]),keff)
# else:
#     for ii in range(len(enrich)):
#         enrichment,splits = problem1.boundaries(enrich[ii],problem=usr_input.problem)
#         print(splits)
#         if usr_input.source is None:
#             problem = eigen_djinn(*problem1.variables(enrich[ii],problem=usr_input.problem),enrich=enrichment,splits=splits,track=usr_input.track,label=labeled)
#             phi,keff = problem.transport(djinn_model,problem=usr_input.problem,multDJ=multDJ)
#             np.save('{}_keff_{}_{:<02}'.format(save_folder,save_file,labels[ii]),keff)
#         else:
#             print('Source Problem')
#             problem = source_djinn(*problem1.variables(enrich[ii],problem=usr_input.problem),enrich=enrichment,splits=splits,label=labeled,track=usr_input.track)
#             phi = problem.transport(djinn_model,problem=usr_input.problem,multDJ=multDJ)    
#         np.save('{}_phi_{}_{:<02}'.format(save_folder,save_file,labels[ii]),phi)
