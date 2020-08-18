#!/usr/bin/env python

import numpy as np
import discrete1.hoth as h
import argparse
import os

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
    enrichment,splits = h.problem.boundaries(enrich[ii],ptype1=usr_input.problem,ptype2=sprob,symm=True)
    if usr_input.source is None:
    	problem = h.data_collect(*h.problem.variables(enrich[ii],ptype=usr_input.problem,symm=True),track=usr_input.track)
    	if usr_input.track == 'power':
	    	phi,keff,track_phi = problem.transport(enrich[ii],problem=usr_input.problem,LOUD=True)	
	    	np.save('mydata/ae_model_data/{}_{:<02}'.format(usr_input.problem,labels[ii]),track_phi)
    	else:
	    	phi,keff = problem.transport(enrich[ii],problem=usr_input.problem,LOUD=True)
    	np.save('mydata/ae_true_1d/phi_{}_{:<02}'.format(usr_input.problem,labels[ii]),phi)
    	np.save('mydata/ae_true_1d/keff_{}_{:<02}'.format(usr_input.problem,labels[ii]),keff)
    else:
    	print('Source Problem')
    	problem = h.source_collect(*h.problem.variables(enrich[ii],ptype=usr_input.problem,symm=True),track=usr_input.track)
    	if usr_input.track == 'source':
	    	phi,track_phi = problem.transport(enrich[ii],problem=usr_input.problem)	
	    	np.save('mydata/ae_source_model_data/{}_{:<02}'.format(usr_input.problem,labels[ii]),track_phi)
    	else:
	    	phi = problem.transport(enrich[ii],problem=usr_input.problem)	
    	np.save('mydata/ae_source_true_1d/phi_{}_{:<02}'.format(usr_input.problem,labels[ii]),phi)
    	
