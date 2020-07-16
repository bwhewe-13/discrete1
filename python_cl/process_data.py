# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 23:48:08 2020

@author: 13wheweb
"""
# Interpolating Data
import numpy as np
# import discrete1.initialize as ex
import discrete1.rogue as r
import glob


# %%
""" Reducing the Data Points """
datatype = 'fission'
adds = np.sort(glob.glob('mydata/djinn_true_1d/{}2*'.format(datatype)))
inds = np.concatenate([np.arange(10),np.arange(10,103,3)])
full = np.linspace(0,55000-550,100,dtype=int)
# complete = np.zeros((2,0,88))
for jj in adds:
    complete = np.zeros((2,0,88))
    data = np.load(jj)
    label = jj.split('2_')[1].split('.')[0]
    print(label)
    count = 0
    for ci,ii in enumerate(full):
        if ci in inds:
            count += 1
            complete = np.hstack((complete,data[:,ii:full[ci+1]]))
    complete = np.hstack((complete,data[:,55000-550:]))
    count += 1
    print(count) # Should be 41
    print(complete.shape)
    np.save('mydata/djinn_true_1d/{}2red_{}'.format(datatype,label),complete)
# %%
""" Interpolating Data Points """
spec = '20'; 
enrich = float(spec)*0.01
direction = 'up'
if direction == 'up':
    end = str(int((enrich+0.05)*100)).zfill(2)
    enrichments = np.round(np.linspace(enrich,enrich+0.04,5),2)
elif direction == 'down':
    end = str(int((enrich-0.05)*100)).zfill(2)
    enrichments = np.round(np.linspace(enrich,enrich-0.04,5),2)
    
start_sca = np.load('mydata/djinn_true_1d/{}2red_{}.npy'.format('scatter',spec))
start_fis = np.load('mydata/djinn_true_1d/{}2red_{}.npy'.format('fission',spec))

complete_fis = np.empty((2,0,88))
complete_sca = np.empty((2,0,88))

_,splits = r.eigen_djinn.boundaries(enrich,symm=True)

for ii in enrichments:
    _,_,_,_,_,scatter,fission,_,_,_ = r.eigen_djinn.variables(ii,symm=True) 
    scatter = scatter[:,0][splits['djinn'][0]][0]
    fission = fission[splits['djinn'][0]][0]
    temp_sca = start_sca.copy()
    temp_fis = start_fis.copy()
    count = 0
    for jj in range(len(start_fis[0])):
        if start_sca[0,jj,0] > 0.01:
            count += 1
            temp_sca[1,jj] = np.append([ii],scatter @ temp_sca[0,jj,1:])
            temp_fis[1,jj] = np.append([ii],fission @ temp_fis[0,jj,1:])
    print(count)
    complete_fis = np.hstack((complete_fis,temp_fis))
    complete_sca = np.hstack((complete_sca,temp_sca))

np.save('mydata/djinn_true_1d/inter2_fission_0{}0{}'.format(spec,end),complete_fis)
np.save('mydata/djinn_true_1d/inter2_scatter_0{}0{}'.format(spec,end),complete_sca)

# %% Compile Data
datatype = 'fission'
adds = np.sort(glob.glob('mydata/djinn_true_1d/inter2_{}*'.format(datatype)))
complete = np.empty((2,0,88))
for ii in adds:
    data = np.load(ii)
    complete = np.hstack((complete,data))
np.save('mydata/djinn_true_1d/inter2_{}_data'.format(datatype),complete)    

datatype = 'scatter'
adds = np.sort(glob.glob('mydata/djinn_true_1d/inter2_{}*'.format(datatype)))
complete = np.empty((2,0,88))
for ii in adds:
    data = np.load(ii)
    complete = np.hstack((complete,data))
np.save('mydata/djinn_true_1d/inter2_{}_data'.format(datatype),complete)    

# %%
adds = np.sort(glob.glob('mydata/djinn_true_1d/scatter2_*'))
complete = np.empty((2,0,88))
for ii in adds:
    data = np.load(ii)
    complete = np.hstack((complete,data))
np.save('mydata/djinn_true_1d/orig2_scatter_data',complete)

# %%
import numpy as np
import glob

# Scatterng Data
# ment = '05'
for ment in ['05','10','15','20','25']:
    address = np.sort(glob.glob('mydata/track_plastic/enrich_{}*'.format(ment)))
    # inds = np.concatenate([np.arange(9),np.arange(9,100,3)]) # for reduce
    complete = []
    for ii,xs in enumerate(address):
        if ii == 0:
            temp = np.load(xs)
            complete.append(temp)
        else:
            temp = np.load(xs)
            inds = np.linspace(0,temp.shape[1],int(temp.shape[1]/450),endpoint=False,dtype=int)
            complete.append(temp[:,inds[-1]:]) # Add last iteration
            # top5 = np.sort(np.concatenate([inds+jj for jj in range(5)]))
            # complete.append(temp[:,top5]) # Add first 5 spatial cells
    full = np.hstack((complete))
    np.save('mydata/track_plastic/enrich_{}_top5'.format(ment),full)
    print(full.shape)
    del temp, full
    
# %%
address = np.sort(glob.glob('mydata/track_plastic/enrich_**_top5.npy'))
complete = [np.load(ii) for ii in address]
full = np.hstack((complete))
np.save('mydata/djinn_true_1d/add_scatter_plastic_data',full)
print(full.shape)


# %%

address = np.sort(glob.glob('mydata/track_scatter/enrich_**_firstlast.npy'))
print(address)
complete = [np.load(ii) for ii in address]
complete = np.hstack((complete))
np.save('mydata/djinn_true_1d/reduced_scatter_data',complete)


# %%

shorty = [np.load(ii) for ii in address]

for ii in range(1,100):
    diff = np.fabs(np.sum(shorty[ii])-np.sum(shorty[ii+1]))
    print(ii,ii+1,diff)



# %%
import glob
import numpy as np

address = np.sort(glob.glob('mydata/track_scatter/enrich_15*'))[:101]
# inds = np.concatenate([np.arange(9),np.arange(9,100,3)]) # for reduce
temp1 = []; temp2 = []
for ii,xs in enumerate(address):
    if ii == 0:
        temp = np.load(xs)
        # temp1.append(temp)
        temp2.append(temp)
    else:
        temp = np.load(xs)
        inds = np.linspace(0,temp.shape[1],int(temp.shape[1]/550),endpoint=False,dtype=int)
        # temp1.append(temp[:,inds[-1]:]) # Add last iteration
        temp2.append(temp[:,inds[-1]:]) # Add last iteration
        skip = inds
        top5 = np.sort(np.concatenate([skip+jj for jj in range(5)]))
        temp2.append(temp[:,top5]) # Add first 5 spatial cells
# original = np.hstack((temp1))
added = np.hstack((temp2))
# %%    

import glob
import numpy as np

address = np.sort(glob.glob('mydata/track_scatter/enrich_15*'))[:101]

full = []
for ii in address:
    full.append(np.load(ii))
original = np.hstack((full))
np.save('mydata/track_scatter/enrich_15_complete',original)
print(original.shape)
