#!/usr/bin/env python

import numpy as np
from discrete1 import correct
from discrete1.setup_ke import problem1,problem2
import argparse

parser = argparse.ArgumentParser(description='Which Enrichments')
parser.add_argument('-enrich',action='store',dest='en',nargs='+')
parser.add_argument('-track',action='store',dest='track')
parser.add_argument('-problem',action='store',dest='problem')
# parser.add_argument('-groups',action='store',dest='groups',nargs='+')
usr_input = parser.parse_args()

dim = 618
# distance = [4,2,4] # 0.89
distance = [5,1.5,3.5]


enrich = [float(jj) for jj in usr_input.en]
labels = [str(jj).split('.')[1] for jj in enrich]

if usr_input.problem == 'pluto':
    for ii in range(len(enrich)):
        enrichment,splits = problem2.boundaries(enrich[ii],problem=usr_input.problem)
        print(splits)
        problem = correct.eigen(*problem2.variables(enrich[ii],dim=dim,distance=distance),enrich=enrichment,track=usr_input.track,splits=splits)
        if usr_input.track == 'power' or usr_input.track == 'both':
            phi,keff,fis_track,sca_track = problem.transport('pluto',enrich=labels[ii],LOUD=True)
            np.save('mydata/model_data_djinn/fission_pluto_full_{:<02}'.format(labels[ii]),fis_track)
            np.save('mydata/model_data_djinn/scatter_pluto_full_{:<02}'.format(labels[ii]),sca_track)
        else:
            phi,keff = problem.transport(usr_input.problem,enrich=labels[ii],LOUD=True)
        np.save('mydata/djinn_pluto/true_phi_{:<02}'.format(labels[ii]),phi)
        np.save('mydata/djinn_pluto/true_keff_{:<02}'.format(labels[ii]),keff)
else:    
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
        # This was uncommented
        # np.save('mydata/track_carbon_djinn/true_phi_{:<02}'.format(labels[ii]),phi)
        # np.save('mydata/track_carbon_djinn/true_keff_{:<02}'.format(labels[ii]),keff)

        np.save('mydata/djinn_{}/true_phi_{:<02}'.format(usr_input.problem,labels[ii]),phi)
        np.save('mydata/djinn_{}/true_keff_{:<02}'.format(usr_input.problem,labels[ii]),keff)




