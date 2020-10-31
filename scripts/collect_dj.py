#!/usr/bin/env python

import numpy as np
from discrete1 import correct
from discrete1.setup import problem1,problem2
import argparse

parser = argparse.ArgumentParser(description='Which Enrichments')
parser.add_argument('-enrich',action='store',dest='en',nargs='+')
parser.add_argument('-track',action='store',dest='track')
parser.add_argument('-problem',action='store',dest='problem')
parser.add_argument('-groups',action='store',dest='groups',nargs='+')
usr_input = parser.parse_args()

# distance = [5,5,10,5,5]
# distance = [2,0.5,4,0.5,2]
distance = [2,1,2]

if usr_input.problem == 'pu':
    groups = [int(jj) for jj in usr_input.groups]
    for gg in range(len(groups)):
        problem = correct.eigen(*problem2.variables(dim=groups[gg],distance=distance),track=usr_input.track)
        phi,keff = problem.transport('pu',0.15)
        np.save('mydata/djinn_pluto/true_phi_{}'.format(str(groups[gg]).zfill(3)),phi)
        np.save('mydata/djinn_pluto/true_keff_{}'.format(str(groups[gg]).zfill(3)),keff)
else:
    enrich = [float(jj) for jj in usr_input.en]
    labels = [str(jj).split('.')[1] for jj in enrich]
    for ii in range(len(enrich)):
        enrichment,splits = problem1.boundaries(enrich[ii],problem=usr_input.problem)
        print(splits)
        problem = correct.eigen(*problem1.variables(enrich[ii],problem=usr_input.problem),enrich=enrichment,track=usr_input.track,splits=splits)
        if usr_input.track == 'power' or usr_input.track == 'both':
            phi,keff,fis_track,sca_track = problem.transport(usr_input.problem,enrich=labels[ii],LOUD=True)
            np.save('mydata/model_data_djinn/fission_{}_full_{:<02}'.format(usr_input.problem,labels[ii]),fis_track)
            np.save('mydata/model_data_djinn/scatter_{}_full_{:<02}'.format(usr_input.problem,labels[ii]),sca_track)
        # elif usr_input.track == 'source':
            # phi,keff = problem.transport(usr_input.problem,enrich=labels[ii],LOUD=True)
            # labels[ii] = 'dj_' + labels[ii] 
            # np.save('mydata/djinn_{}/phi_{:<02}'.format(usr_input.problem,labels[ii]),phi)
            # np.save('mydata/djinn_{}/keff_{:<02}'.format(usr_input.problem,labels[ii]),keff)
        else:
            phi,keff = problem.transport(usr_input.problem,enrich=labels[ii],LOUD=True)

        # phi,keff = problem.transport(usr_input.problem,enrich=labels[ii],LOUD=True)
        np.save('mydata/djinn_{}/true_phi_{:<02}'.format(usr_input.problem,labels[ii]),phi)
        np.save('mydata/djinn_{}/true_keff_{:<02}'.format(usr_input.problem,labels[ii]),keff)




