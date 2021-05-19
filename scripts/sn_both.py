#!/usr/bin/env python

import numpy as np
from discrete1.critical import Critical
import argparse, os
import time

parser = argparse.ArgumentParser(description='Enrichment')

parser.add_argument('-enrich',action='store',dest='en',nargs='+')
parser.add_argument('-fission',action='store',dest='fission',nargs='+')
parser.add_argument('-refl',action='store',dest='refl',nargs='+')
parser.add_argument('-fuel',action='store',dest='fuel',nargs='+')
parser.add_argument('-time',action='store',dest='its')
parser.add_argument('-iteration',action='store',dest='iters')
usr_input = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Enrichment and associated labels
enrich = [float(jj) for jj in usr_input.en]
labels = [str(jj).split('.')[1] for jj in enrich]

labeled = False
if usr_input.fission[0] == 'label':
    labeled = True

# lists --> [[scatter/fuel, fission/fuel],[scatter/reflector, fission/reflector]]

# Fission Fuel
fission_djinn_model = 'fission_1d/pluto/djinn_' + usr_input.fission[0] + '/model_' + usr_input.fission[1]
fission_auto_model = 'autoencoder/pluto_fission/model_300-150_' + usr_input.fission[0]

# Scatter Fuel
fuel_djinn_model = 'scatter_1d/pluto/djinn_' + usr_input.fuel[0] + '/model_' + usr_input.fuel[1]
fuel_auto_model = 'autoencoder/pluto_scatter/model_300-150_' + usr_input.fuel[0]

# Scatter Refl
refl_djinn_model = 'scatter_1d/hdpe618/djinn_' + usr_input.refl[0] + '/model_' + usr_input.refl[1]
refl_auto_model = 'autoencoder/hdpe_scatter/model_300-150_' + usr_input.refl[0]

# Combine
dj_model = [[fuel_djinn_model,fission_djinn_model],refl_djinn_model]
ae_model = [[fuel_auto_model,fission_auto_model],refl_auto_model]


save_folder = 'mydata/djinn_pluto/both_{}_{}_{}'.format(usr_input.fission[1],usr_input.fuel[1],usr_input.refl[1])
save_file = usr_input.fission[0]


atype = 'both'

if usr_input.its:
    its = int(usr_input.its)
    for ii in range(len(enrich)):
        timer = []
        for tt in range(its):
            start = time.time()
            _,_ = Critical.run_djae('pu',enrich[ii],dj_model,ae_model,atype,double=True,label=labeled)
            end = time.time()
            timer.append(end-start)
        np.save('{}_time_{}_{}_{:<02}'.format(save_folder,save_file,usr_input.iters,labels[ii]),timer)
else:
    for ii in range(len(enrich)):
        phi,keff = Critical.run_djae('pu',enrich[ii],dj_model,ae_model,atype,double=True,label=labeled) # focus=usr_input.focus,
        #np.save('{}_phi_{}_{:<02}'.format(save_folder,save_file,labels[ii]),phi)
        #np.save('{}_keff_{}_{:<02}'.format(save_folder,save_file,labels[ii]),keff)

