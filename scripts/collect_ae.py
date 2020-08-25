#!/usr/bin/env python

import numpy as np
import os, argparse
import discrete1.correct as c
import discrete1.setup as s

parser = argparse.ArgumentParser(description='Enrichment')
parser.add_argument('-enrich',action='store',dest='en',nargs='+') #Enrichment looking into
parser.add_argument('-problem',action='store',dest='problem') # Problem set up
parser.add_argument('-track',action='store',dest='track')
parser.add_argument('-source',action='store',dest='source')
usr_input = parser.parse_args()

enrich = [float(jj) for jj in usr_input.en]
labels = [str(jj).split('.')[1] for jj in enrich]


for ii in range(len(enrich)):
    enrichment,splits = s.problem1.boundaries(enrich[ii],problem=usr_input.problem)
    print(splits)
    if usr_input.source is None:
        setup = c.eigen_collect(*s.problem1.variables(enrich[ii],probelm=usr_input.problem),track=usr_input.track)
        if usr_input.track == 'power':
            phi,keff,track_phi = setup.transport(enrich[ii],problem=usr_input.problem,LOUD=True)  
            np.save('mydata/ae_model_data/{}_{:<02}'.format(usr_input.problem,labels[ii]),track_phi)
        else:
            phi,keff = problem.transport(enrich[ii],problem=usr_input.problem,LOUD=True)
        np.save('mydata/ae_true_1d/phi_{}_{:<02}'.format(usr_input.problem,labels[ii]),phi)
        np.save('mydata/ae_true_1d/keff_{}_{:<02}'.format(usr_input.problem,labels[ii]),keff)
    else:
        print('Source Problem')
        setup = c.source(*s.problem1.variables(enrich[ii],problem=usr_input.problem),track=usr_input.track,enrich=enrichment,splits=splits)
        if usr_input.track == 'source':
            phi,track_fmult,track_smult = setup.transport(enrich[ii],problem=usr_input.problem) 
            np.save('mydata/ae_source_model_data/smult_{}{:<02}'.format(usr_input.problem,labels[ii]),track_smult)
            np.save('mydata/ae_source_model_data/fmult_{}{:<02}'.format(usr_input.problem,labels[ii]),track_fmult)
        else:
            phi = problem.transport(enrich[ii],problem=usr_input.problem)   
        np.save('mydata/ae_source_true_1d/phi_{}_{:<02}'.format(usr_input.problem,labels[ii]),phi)
        
