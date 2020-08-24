#!/usr/bin/env python

import numpy as np
import discrete1.theTruth as truth
import discrete1.theProcess as pro
import argparse

parser = argparse.ArgumentParser(description='Which Enrichments')
parser.add_argument('-enrich',action='store',dest='en',nargs='+')
parser.add_argument('-track',action='store',dest='track')
parser.add_argument('-problem',action='store',dest='problem')
usr_input = parser.parse_args()

enrich = [float(jj) for jj in usr_input.en]
labels = [str(jj).split('.')[1] for jj in enrich]

prob_scat = usr_input.problem + '_full'

for ii in range(len(enrich)):
    enrichment,splits = pro.problem.boundaries(enrich[ii],ptype1=usr_input.problem,ptype2=prob_scat,symm=True)
    print(splits)
    problem = truth.eigen(*pro.problem.variables(enrich[ii],ptype=usr_input.problem,symm=True),track=usr_input.track,enrich=enrichment,splits=splits)
    if usr_input.track is None:
        phi,keff = problem.transport(usr_input.problem,enrich=labels[ii],LOUD=True)
    else:
        phi,keff,fis_track,sca_track = problem.transport(usr_input.problem,enrich=labels[ii],LOUD=True)
        np.save('mydata/model_data_djinn/fission_{}_full_{:<02}'.format(usr_input.problem,labels[ii]),fis_track)
        np.save('mydata/model_data_djinn/scatter_{}_full_{:<02}'.format(usr_input.problem,labels[ii]),sca_track)
    np.save('mydata/djinn_{}/phi_{:<02}'.format(usr_input.problem,labels[ii]),phi)
    np.save('mydata/djinn_{}/keff_{:<02}'.format(usr_input.problem,labels[ii]),keff)




