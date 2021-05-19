#!/usr/bin/env python

import numpy as np
from discrete1.critical import Critical
import argparse, os
import time

parser = argparse.ArgumentParser(description='Enrichment')

parser.add_argument('-enrich',action='store',dest='en',nargs='+')
parser.add_argument('-time',action='store',dest='its')
parser.add_argument('-iteration',action='store',dest='iters')
usr_input = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Enrichment and associated labels
enrich = [float(jj) for jj in usr_input.en]
labels = [str(jj).split('.')[1] for jj in enrich]

# lists --> [[scatter/fuel, fission/fuel],[scatter/reflector, fission/reflector]]

save_folder = 'mydata/djinn_pluto/true'

if usr_input.its:
    its = int(usr_input.its)
    for ii in range(len(enrich)):
        timer = []
        for tt in range(its):
            start = time.time()
            _,_ = Critical.run('pu',enrich[ii])
            end = time.time()
            timer.append(end-start)
        np.save('{}_time_{}_{:<02}'.format(save_folder,usr_input.iters,labels[ii]),timer)
else:
    for ii in range(len(enrich)):
        _,_ = Critical.run('pu',enrich[ii]) # focus=usr_input.focus,
        #np.save('{}_phi_{}_{:<02}'.format(save_folder,save_file,labels[ii]),phi)
        #phi,keff = Critical.run_djae('pu',enrich[ii],dj_model,ae_model,atype,double=True,label=labeled) # focus=usr_input.focus,
        #np.save('{}_keff_{}_{:<02}'.format(save_folder,save_file,labels[ii]),keff)

