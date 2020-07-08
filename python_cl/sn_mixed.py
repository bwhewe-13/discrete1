#!/usr/bin/env python

import numpy as np
import discrete1.rogue as r
from discrete1.util import nnets
import argparse

parser = argparse.ArgumentParser(description='Enrichment')
parser.add_argument('-xs',action='store',dest='xs') # Which matrix multiply DJINN estimates, 'fission','scatter', or 'both'
usr_input = parser.parse_args()
process = None

if usr_input.xs == 'both':
    print('Both Reg')
    file1 = 'both'; file3 = '_reg'; label=False
    djinn_model = []    
    nums = nnets.djinn_metric('{}_1d/djinn{}/error/model*'.format('scatter',file3),clean=True)
    djinn_model.append('{}_1d/djinn{}/model_{}'.format('scatter',file3,nums))
    nums = nnets.djinn_metric('{}_1d/djinn{}/error/model*'.format('fission',file3),clean=True)
    djinn_model.append('{}_1d/djinn{}/model_{}'.format('fission',file3,nums))
    print('DJINN Model',djinn_model)
    enrichment,splits = r.eigen_djinn.boundaries_mixed(0.12,symm=True)
    problem = r.eigen_djinn_symm(*r.eigen_djinn.variables_mixed(symm=True),dtype=file1,enrich=enrichment,splits=splits,label=label)
    phi,keff = problem.transport(djinn_model,process=process,LOUD=True,MAX_ITS=10)
    np.save('mydata/djinn_{}_1d/phi_{}mixed1'.format(file1,file3),phi)
    np.save('mydata/djinn_{}_1d/keff_{}mixed1'.format(file1,file3),keff)

    print('Both Label')
    file1 = 'both'; file3 = '_label'; label=True
    djinn_model = []    
    nums = nnets.djinn_metric('{}_1d/djinn{}/error/model*'.format('scatter',file3),clean=True)
    djinn_model.append('{}_1d/djinn{}/model_{}'.format('scatter',file3,nums))
    nums = nnets.djinn_metric('{}_1d/djinn{}/error/model*'.format('fission',file3),clean=True)
    djinn_model.append('{}_1d/djinn{}/model_{}'.format('fission',file3,nums))
    print('DJINN Model',djinn_model)
    enrichment,splits = r.eigen_djinn.boundaries_mixed(0.12,symm=True)
    problem = r.eigen_djinn_symm(*r.eigen_djinn.variables_mixed(symm=True),dtype=file1,enrich=enrichment,splits=splits,label=label)
    phi,keff = problem.transport(djinn_model,process=process,LOUD=True,MAX_ITS=10)
    np.save('mydata/djinn_{}_1d/phi_{}mixed1'.format(file1,file3),phi)
    np.save('mydata/djinn_{}_1d/keff_{}mixed1'.format(file1,file3),keff)

elif usr_input.xs == 'fission':
    print('Fission Reg')
    file1 = 'fission'; file3 = '_reg'; label=False
    nums = nnets.djinn_metric('{}_1d/djinn{}/error/model*'.format(file1,file3),clean=True)
    djinn_model = '{}_1d/djinn{}/model_{}'.format(file1,file3,nums)
    print('DJINN Model',djinn_model)
    enrichment,splits = r.eigen_djinn.boundaries_mixed(0.12,symm=True)
    problem = r.eigen_djinn_symm(*r.eigen_djinn.variables_mixed(symm=True),dtype=file1,enrich=enrichment,splits=splits,label=label)
    phi,keff = problem.transport(djinn_model,process=process,LOUD=True)
    np.save('mydata/djinn_{}_1d/phi_{}mixed1'.format(file1,file3),phi)
    np.save('mydata/djinn_{}_1d/keff_{}mixed1'.format(file1,file3),keff)

    print('Fission Label')
    file1 = 'fission'; file3 = '_label'; label=True
    nums = nnets.djinn_metric('{}_1d/djinn{}/error/model*'.format(file1,file3),clean=True)
    djinn_model = '{}_1d/djinn{}/model_{}'.format(file1,file3,nums)
    print('DJINN Model',djinn_model)
    enrichment,splits = r.eigen_djinn.boundaries_mixed(0.12,symm=True)
    problem = r.eigen_djinn_symm(*r.eigen_djinn.variables_mixed(symm=True),dtype=file1,enrich=enrichment,splits=splits,label=label)
    phi,keff = problem.transport(djinn_model,process=process,LOUD=True)
    np.save('mydata/djinn_{}_1d/phi_{}mixed1'.format(file1,file3),phi)
    np.save('mydata/djinn_{}_1d/keff_{}mixed1'.format(file1,file3),keff)

elif usr_input.xs == 'scatter':
    print('Scatter Reg')
    file1 = 'scatter'; file3 = '_reg'; label=False
    nums = nnets.djinn_metric('{}_1d/djinn{}/error/model*'.format(file1,file3),clean=True)
    djinn_model = '{}_1d/djinn{}/model_{}'.format(file1,file3,nums)
    print('DJINN Model',djinn_model)
    enrichment,splits = r.eigen_djinn.boundaries_mixed(0.12,symm=True)
    problem = r.eigen_djinn_symm(*r.eigen_djinn.variables_mixed(symm=True),dtype=file1,enrich=enrichment,splits=splits,label=label)
    phi,keff = problem.transport(djinn_model,process=process,LOUD=True)
    np.save('mydata/djinn_{}_1d/phi_{}mixed1'.format(file1,file3),phi)
    np.save('mydata/djinn_{}_1d/keff_{}mixed1'.format(file1,file3),keff)

    print('Scatter Label')
    file1 = 'scatter'; file3 = '_label'; label=True
    nums = nnets.djinn_metric('{}_1d/djinn{}/error/model*'.format(file1,file3),clean=True)
    djinn_model = '{}_1d/djinn{}/model_{}'.format(file1,file3,nums)
    print('DJINN Model',djinn_model)
    enrichment,splits = r.eigen_djinn.boundaries_mixed(0.12,symm=True)
    problem = r.eigen_djinn_symm(*r.eigen_djinn.variables_mixed(symm=True),dtype=file1,enrich=enrichment,splits=splits,label=label)
    phi,keff = problem.transport(djinn_model,process=process,LOUD=True)
    np.save('mydata/djinn_{}_1d/phi_{}mixed1'.format(file1,file3),phi)
    np.save('mydata/djinn_{}_1d/keff_{}mixed1'.format(file1,file3),keff)