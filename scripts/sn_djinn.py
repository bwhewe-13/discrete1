#!/usr/bin/env python

import numpy as np
from discrete1.critical import Critical
import argparse, os
import time
import datetime

parser = argparse.ArgumentParser(description='Enrichment')

parser.add_argument('-enrich',action='store',dest='en',nargs='+')
parser.add_argument('-time',action='store',dest='its')
parser.add_argument('-iteration',action='store',dest='iters')
parser.add_argument('-reduce',action='store',dest='reduce')
usr_input = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

day = datetime.datetime.now().strftime('%x').replace('/','-')
print('Today is ',day)

# Enrichment and associated labels
enrich = [float(jj) for jj in usr_input.en]
labels = [str(jj).split('.')[1] for jj in enrich]

# lists --> [[scatter/fuel, fission/fuel],[scatter/reflector, fission/reflector]]

save_folder = 'mydata/djinn_pluto/true'

if usr_input.its:
    key = np.array(['day','time','type','elapsed','enrichment'])
    ktype = 'Reduce' if usr_input.reduce else 'Full'
    its = int(usr_input.its)
    for ii in range(len(enrich)):
        # timer = []
        for tt in range(its):
            start = time.time()
            hour = datetime.datetime.now().strftime('%X')
            if usr_input.reduce:
                _,_ = Critical.run_reduce('pu',enrich[ii])
            else:
                _,_ = Critical.run('pu',enrich[ii])
            end = time.time()
            # timer.append(end-start)
            key = np.vstack((key,[day,hour,ktype,end-start,labels[ii]]))
        # np.save('{}_time_{}_{:<02}'.format(save_folder,usr_input.iters,labels[ii]),timer)
    # Add to key
    full_key_name = 'true_reduce_time.npy' if usr_input.reduce else 'true_full_time.npy'
    full_key = np.load(full_key_name)
    full_key = np.vstack((full_key,key[1:]))
    np.save(full_key_name,full_key)
else:
    for ii in range(len(enrich)):
        _,_ = Critical.run('pu',enrich[ii]) # focus=usr_input.focus,
        #np.save('{}_phi_{}_{:<02}'.format(save_folder,save_file,labels[ii]),phi)
        #phi,keff = Critical.run_djae('pu',enrich[ii],dj_model,ae_model,atype,double=True,label=labeled) # focus=usr_input.focus,
        #np.save('{}_keff_{}_{:<02}'.format(save_folder,save_file,labels[ii]),keff)

