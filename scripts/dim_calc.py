#!/usr/bin/env python

import numpy as np
from discrete1 import correct
from discrete1.setup import problem1
import argparse

parser = argparse.ArgumentParser(description='Which Enrichments')
parser.add_argument('-enrich',action='store',dest='enrich',nargs='+')
parser.add_argument('-distance',action='store',dest='distance',nargs='+')
parser.add_argument('-problem',action='store',dest='problem')
usr_input = parser.parse_args()

enrich = [float(jj) for jj in usr_input.enrich]
labels = [str(jj).split('.')[1] for jj in enrich]

if usr_input.distance is None:
    distance = [45,35,20]
    dist_label = '45'
else:
    distance = [int(kk) for kk in usr_input.distance]
    dist_label = str(distance[0]).zfill(2)

for ii in range(len(enrich)):
    enrichment,splits = problem1.boundaries(enrich[ii],problem=usr_input.problem,distance=distance)
    print(splits)
    problem = correct.eigen(*problem1.variables(enrich[ii],problem=usr_input.problem,distance=distance),enrich=enrichment,splits=splits)
    phi,keff = problem.transport(usr_input.problem,enrich=labels[ii],LOUD=True)
    np.save('mydata/svd_{}_real/hdpe_{}/true_phi_{:<02}'.format(usr_input.problem,dist_label,labels[ii]),phi)
    np.save('mydata/svd_{}_real/hdpe_{}/true_keff_{:<02}'.format(usr_input.problem,dist_label,labels[ii]),keff)
    np.save('mydata/svd_{}_15/hdpe_{}/true_phi_{:<02}'.format(usr_input.problem,dist_label,labels[ii]),phi)
    np.save('mydata/svd_{}_15/hdpe_{}/true_keff_{:<02}'.format(usr_input.problem,dist_label,labels[ii]),keff)