#!/usr/bin/env python

import numpy as np
import discrete1.slab as s
import discrete1.initialize as ex
import argparse

parser = argparse.ArgumentParser(description='Which Enrichments')
parser.add_argument('-e',action='store',dest='en',type=int)
usr_input = parser.parse_args()

allmat_fis = []
allmat_sca = []
enrich1 = [0.0,0.05]
enrich2 = [0.10,0.15]
enrich3 = [0.20,0.25]

if usr_input.en == 1:
    enrich = enrich1
elif usr_input.en == 2:
    enrich = enrich2
elif usr_input.en == 3:
    enrich = enrich3
print(enrich)

labels = [str(jj).split('.')[1] for jj in enrich]
for ii in range(len(enrich)):
    enrichment,splits = ex.eigen_djinn.boundaries(enrich[ii],symm=True)
    problem = s.eigen_symm(*ex.eigen_djinn.variables(enrich[ii],symm=True),track='both',enrich=enrichment,splits=splits)
    #problem = s.eigen(*ex.eigen_djinn.variables(enrich[ii]))
    phi,keff,fis_track,sca_track = problem.transport(LOUD=True)
    allmat_fis.append(fis_track)
    allmat_sca.append(sca_track)
    np.save('mydata/djinn_true_1d/true_phi_{:<02}'.format(labels[ii]),phi)
    np.save('mydata/djinn_true_1d/true_keff_{:<02}'.format(labels[ii]),keff)
    np.save('mydata/djinn_true_1d/true_track_fission_{:<02}'.format(labels[ii]),fis_track)
    np.save('mydata/djinn_true_1d/true_track_scatter_{:<02}'.format(labels[ii]),sca_track)
    # Send to Dropbox
    #np.save('../../Dropbox/data_collection_1d/true_phi_{:<02}'.format(labels[ii]),phi)
    #np.save('../../Dropbox/data_collection_1d/true_keff_{:<02}'.format(labels[ii]),keff)
    #np.save('../../Dropbox/data_collection_1d/true_track_fission_{:<02}'.format(labels[ii]),fis_track)
    #np.save('../../Dropbox/data_collection_1d/true_track_scatter_{:<02}'.format(labels[ii]),sca_track)

#np.save('mydata/djinn_true_1d/all_tracking_fission',allmat_fis)
#np.save('../../Dropbox/data_collection_1d/all_tracking_fission',allmat_fis)
#np.save('mydata/djinn_true_1d/all_tracking_scatter',allmat_sca)
#np.save('../../Dropbox/data_collection_1d/all_tracking_scatter',allmat_sca)

#combined_fis = np.zeros((1,3,87))
#for ii in range(len(allmat_fis)):
#    combined_fis = np.vstack((combined_fis,allmat_fis[ii]))
#combined_fis = combined_fis[1:]
#np.save('mydata/djinn_true_1d/true_track_fission_combined',combined_fis)
#np.save('../../Dropbox/data_collection_1d/true_track_fission_combined',combined_fis)

#combined_sca = np.zeros((1,3,87))
#for jj in range(len(allmat_sca)):
#    combined_sca = np.vstack((combined_sca,allmat_sca[jj]))	
#combined_sca = combined_sca[1:]
#np.save('mydata/djinn_true_1d/true_track_scatter_combined',combined_sca)
#np.save('../../Dropbox/data_collection_1d/true_track_scatter_combined',combined_sca)

