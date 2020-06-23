#!/usr/bin/env python

import numpy as np
import discrete1.initialize as ex
import argparse

parser = argparse.ArgumentParser(description='Enrichment')
#parser.add_argument('-enrich',action='store',dest='enrich',nargs='+') #Enrichment looking into
parser.add_argument('-interest',action='store',dest='interest')
parser.add_argument('-top',action='store',dest='top')
usr_input = parser.parse_args()

spec = float(usr_input.interest)
label = str(spec*100).split('.')[0].zfill(2)
if usr_input.top:
    enrichments = np.round(np.linspace(spec-0.04,spec-0.01,4),2)
    label2 = str((spec-0.05)*100).split('.')[0].zfill(2)
else:
    enrichments = np.round(np.linspace(spec+0.01,spec+0.04,4),2)
    label2 = str((spec+0.05)*100).split('.')[0].zfill(2)

start_fis = np.load('mydata/djinn_true_1d/orig_fission_{}.npy'.format(label))
start_sca = np.load('mydata/djinn_true_1d/orig_scatter_{}.npy'.format(label))
complete_fis = np.empty((0,3,87))
complete_sca = np.empty((0,3,87))

_,splits = ex.eigen_djinn.boundaries(float(usr_input.interest),symm=True)
print(label,enrichments,label2)
count = 0
for ii in enrichments:
    _,_,_,_,_,scatter,fission,_,_,_ = ex.eigen_djinn.variables(ii,symm=True) 
    scatter = scatter[:,0][splits['djinn'][0]][0]
    fission = fission[splits['djinn'][0]][0]
    temp_sca = start_sca.copy()
    temp_fis = start_fis.copy()
    for jj in range(len(start_fis)):
        if sum(start_sca[jj,0]) != 0:
            temp_sca[jj,2] = scatter @ temp_sca[jj,1]
            temp_fis[jj,2] = fission @ temp_fis[jj,1]
    complete_fis = np.vstack((complete_fis,temp_fis))
    complete_sca = np.vstack((complete_sca,temp_sca))
#print(complete_sca.shape)

np.save('mydata/djinn_true_1d/inter_fission_0{}0{}'.format(label,label2),complete_fis)
np.save('mydata/djinn_true_1d/inter_scatter_0{}0{}'.format(label,label2),complete_sca)
#else:
#    np.save('mydata/djinn_true_1d/inter_fission_0{}0{}'.format(label,label2),complete_fis)
#    np.save('mydata/djinn_true_1d/inter_scatter_0{}0{}'.format(label,label2),complete_sca)