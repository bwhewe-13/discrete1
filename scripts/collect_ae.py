#!/usr/bin/env python

import numpy as np
import os, argparse
import discrete1.theTruth as truth
import discrete1.theProcess as pro

parser = argparse.ArgumentParser(description='Enrichment')
parser.add_argument('-enrich',action='store',dest='en',nargs='+') #Enrichment looking into
parser.add_argument('-problem',action='store',dest='problem') # Problem set up
parser.add_argument('-track',action='store',dest='track')
parser.add_argument('-source',action='store',dest='source')
usr_input = parser.parse_args()

enrich = [float(jj) for jj in usr_input.en]
labels = [str(jj).split('.')[1] for jj in enrich]

sprob = '{}_full'.format(usr_input.problem)

for ii in range(len(enrich)):
    enrichment,splits = pro.problem.boundaries(enrich[ii],ptype1=usr_input.problem,ptype2=sprob,symm=True)
    if usr_input.source is None:
        problem = truth.eigen_collect(*pro.problem.variables(enrich[ii],ptype=usr_input.problem,symm=True),track=usr_input.track)
        if usr_input.track == 'power':
            phi,keff,track_phi = problem.transport(enrich[ii],problem=usr_input.problem,LOUD=True)  
            np.save('mydata/ae_model_data/{}_{:<02}'.format(usr_input.problem,labels[ii]),track_phi)
        else:
            phi,keff = problem.transport(enrich[ii],problem=usr_input.problem,LOUD=True)
        np.save('mydata/ae_true_1d/phi_{}_{:<02}'.format(usr_input.problem,labels[ii]),phi)
        np.save('mydata/ae_true_1d/keff_{}_{:<02}'.format(usr_input.problem,labels[ii]),keff)
    else:
        print('Source Problem')
        problem = truth.source(*pro.problem.variables(enrich[ii],ptype=usr_input.problem,symm=True),track=usr_input.track,enrich=enrichment,splits=splits)
        if usr_input.track == 'source':
            phi,track_smult,track_fmult = problem.transport(enrich[ii],problem=usr_input.problem) 
            np.save('mydata/ae_source_model_data/smult_{}{:<02}'.format(usr_input.problem,labels[ii]),track_smult)
            np.save('mydata/ae_source_model_data/fmult_{}{:<02}'.format(usr_input.problem,labels[ii]),track_fmult)
        else:
            phi = problem.transport(enrich[ii],problem=usr_input.problem)   
        np.save('mydata/ae_source_true_1d/phi_{}_{:<02}'.format(usr_input.problem,labels[ii]),phi)
        
