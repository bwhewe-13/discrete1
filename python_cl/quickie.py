#!/usr/bin/env python

import numpy as np
import discrete1.rogue as r
import argparse

parser = argparse.ArgumentParser(description='Which Enrichments')
parser.add_argument('-e',action='store',dest='en',type=int)
usr_input = parser.parse_args()

enrich1 = [0.05]
enrich2 = [0.10] #,0.15]
enrich3 = [0.15]
#enrich3 = [0.20,0.25]

if usr_input.en == 1:
    enrich = enrich1
elif usr_input.en == 2:
    enrich = enrich2
elif usr_input.en == 3:
    enrich = enrich3
print(enrich)

labels = [str(jj).split('.')[1] for jj in enrich]
for ii in range(len(enrich)):
    enrichment,splits = r.eigen_djinn.boundaries(enrich[ii],symm=True)
    problem = r.eigen_symm(*r.eigen_djinn.variables(enrich[ii],symm=True),track='both',enrich=enrichment,splits=splits['djinn'])
    #problem = s.eigen(*ex.eigen_djinn.variables(enrich[ii]))
    phi,keff,fis_track,sca_track = problem.transport(LOUD=True)
    np.save('mydata/djinn_true_1d/phi2_{:<02}'.format(labels[ii]),phi)
    np.save('mydata/djinn_true_1d/keff2_{:<02}'.format(labels[ii]),keff)
    np.save('mydata/djinn_true_1d/fission2_{:<02}'.format(labels[ii]),fis_track)
    np.save('mydata/djinn_true_1d/scatter2_{:<02}'.format(labels[ii]),sca_track)

