#!/usr/bin/env python

import numpy as np
import discrete1.rogue as r
import argparse

parser = argparse.ArgumentParser(description='Which Enrichments')
parser.add_argument('-enrich',action='store',dest='en',nargs='+')
parser.add_argument('-track',action='store',dest='track')
usr_input = parser.parse_args()

enrich = [float(jj) for jj in usr_input.en]
labels = [str(jj).split('.')[1] for jj in enrich]
for ii in range(len(enrich)):
    enrichment,splits = r.eigen_djinn.boundaries(enrich[ii],symm=True)
    # print('{:<02}'.format(str(np.amax(enrichment)).split('.')[1]))
    problem = r.eigen_symm(*r.eigen_djinn.variables(enrich[ii],symm=True),track=usr_input.track,enrich=enrichment,splits=splits['keep'])
    if usr_input.track is None:
        phi,keff = problem.transport(LOUD=True)
    else:
        phi,keff = problem.transport(LOUD=True)
        # phi,keff,fis_track,sca_track = problem.transport(LOUD=True)
        # np.save('mydata/djinn_true_1d/fission2_{:<02}'.format(labels[ii]),fis_track)
        # np.save('mydata/djinn_true_1d/scatter_trackFull_{:<02}'.format(labels[ii]),sca_track)
    # np.save('mydata/djinn_true_1d/phi_{:<02}'.format(labels[ii]),phi)
    # np.save('mydata/djinn_true_1d/keff_{:<02}'.format(labels[ii]),keff)


# enrichment,splits = r.eigen_djinn.boundaries_noplastic(0.15,symm=True)
# print(splits)
# problem = r.eigen_symm(*r.eigen_djinn.variables_noplastic(symm=True),enrich=enrichment,splits=splits['djinn'])
# phi,keff = problem.transport(LOUD=True)
# np.save('mydata/djinn_true_1d/phi_noplastic_15',phi)
# np.save('mydata/djinn_true_1d/keff_noplastic_15',keff)