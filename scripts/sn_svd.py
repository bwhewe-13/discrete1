#!/usr/bin/env python

import numpy as np
from discrete1.svd_prob import eigen
from discrete1.setup import problem1
import argparse, os, glob

parser = argparse.ArgumentParser(description='Enrichment')

parser.add_argument('-problem',action='store',dest='problem') 
parser.add_argument('-source',action='store',dest='source')
parser.add_argument('-original',action='store',dest='original')
parser.add_argument('-enrich',action='store',dest='enrich',nargs='+') 
parser.add_argument('-rank',action='store',dest='rank',nargs='+') 
usr_input = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Enrichment and associated labels
enrich = [float(kk) for kk in usr_input.enrich]
enrich_labels = [str(kk).split('.')[1] for kk in enrich]
# Rank and associated labels
rank = [int(kk) for kk in usr_input.rank]
rank_labels = [str(kk).zfill(2) for kk in rank]

file2 = '15'
solutions = [None for ii in range(len(enrich_labels))]
if usr_input.original == 'True':
	file2 = 'real'
	adds = np.sort(glob.glob('mydata/djinn_carbon/true_phi_'+('[0-2][0,5]')+'.npy'))
	print(adds)
	solutions = [np.load(kk) for kk in adds]

save_folder = 'mydata/svd_{}_{}/'.format(usr_input.problem,file2)

for ii in range(len(enrich)):
    problem = eigen(*problem1.variables(enrich[ii],problem=usr_input.problem))
    for jj in range(len(rank)):
        phi,keff = problem.transport(problem=usr_input.problem,rank=rank[jj],solution=solutions[ii])
        np.save('{}rank{}_keff_{:<02}'.format(save_folder,rank_labels[jj],enrich_labels[ii]),keff)
        np.save('{}rank{}_phi_{:<02}'.format(save_folder,rank_labels[jj],enrich_labels[ii]),phi)
