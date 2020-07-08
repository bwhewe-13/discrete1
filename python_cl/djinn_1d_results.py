import numpy as np
import matplotlib.pyplot as plt
import glob, re
from djinn import djinn
from discrete1.util import display,nnets,sn
import discrete1.initialize as ex

# %% DJINN Models
# Scatter Models
# nums = nnets.djinn_metric('scatter_enrich_1d_djinn_model/error/model*',clean=True)
# dj1 = 'scatter_enrich_1d_djinn_model/model_{}'.format(nums)
# dj_scatter_enrich = djinn.load(model_name=dj1)
# nums = nnets.djinn_metric('scatter_1d_djinn_model/error/model*',clean=True)
# dj2 = 'scatter_1d_djinn_model/model_{}'.format(nums)
# dj_scatter = djinn.load(model_name=dj2)
# Fission Models
nums = nnets.djinn_metric('fission_enrich_1d_djinn_model/error/model*',clean=True)
dj3 = 'fission_enrich_1d_djinn_model/model_{}'.format(nums)
dj_fission_enrich = djinn.load(model_name=dj3)
nums = nnets.djinn_metric('fission_1d_djinn_model/error/model*',clean=True)
dj4 = 'fission_1d_djinn_model/model_{}'.format(nums)
dj_fission = djinn.load(model_name=dj4)

# %% Loading Data
enrich = 0.05; 
label = f'{enrich:.2f}'.split('.')[1]
# Load in True phi - 5%, 10%, 15%, 20%, 25%
phi = np.load('mydata/djinn_true_1d/phi_{}.npy'.format(label))[:,0]
_,_,_,_,_,scatter,fission,_,_,_ = ex.eigen_djinn.variables(enrich,symm=True)
enrichments,splits = ex.eigen_djinn.boundaries(enrich,symm=True)
fissionphi = sn.cat(np.einsum('ijk,ik->ij',fission,phi),splits['djinn'])
scatterphi = sn.cat(np.einsum('ijk,ik->ij',scatter[:,0],phi),splits['djinn'])

# %% Predict Matrix Multiply
pred1 = sn.cat(np.concatenate((np.expand_dims(enrichments,axis=1),phi),axis=1),splits['djinn'])
pred2 = sn.cat(phi,splits['djinn'])

djpred_fission_enrich = dj_fission_enrich.predict(pred1)
djpred_fission = dj_fission.predict(pred2)
# djpred_scatter_enrich = dj_scatter_enrich.predict(pred1)
# djpred_scatter = dj_scatter.predict(pred2)

# %% Space
energy = display.gridPlot(np.load('mydata/uh3_20_enriched/energyGrid.npy'))
space = np.linspace(45.1,100,550)
spot = -1
plt.semilogx(energy,fissionphi[spot],label='True',c='k',ls='--')
plt.semilogx(energy,djpred_fission[spot],label='DJINN',alpha=0.5,c='b')
plt.semilogx(energy,djpred_fission_enrich[spot],label='DJINN Enrich',alpha=0.5,c='r')
plt.legend(loc='best'); plt.grid()
plt.xlabel('Energy (eV)'); plt.ylabel('Magnitude of Matrix Multiply')
plt.title('Location {} cm'.format(space[spot]))
plt.show()

# %% Energy
en = 42
plt.plot(space,fissionphi[:,en],label='True',c='k',ls='--')
plt.plot(space,djpred_fission[:,en],label='DJINN',alpha=0.5,c='b')
plt.plot(space,djpred_fission_enrich[:,en],label='DJINN Enrich',alpha=0.5,c='r')
plt.legend(loc='best'); plt.grid()
plt.xlabel('Spatial Cell'); plt.ylabel('Magnitude of Matrix Multiply')
plt.title('Energy Level {} eV'.format(np.round(energy[en],4)))
plt.show()







# %% Data Collection - Small group
import numpy as np
import discrete1.initialize as ex
import matplotlib.pyplot as plt
from discrete1.util import display
_,splits = ex.eigen_djinn.boundaries(0.05,symm=True)
# True Results
look = '15'
phi = np.load('mydata/djinn_true_1d/phi2_{}.npy'.format(look))
keff = np.load('mydata/djinn_true_1d/keff2_{}.npy'.format(look))[-1]
# Fission DJINN - labeled and unlabeled
phi_fiss1 = np.load('mydata/djinn_fission_1d/phi_small_norm_reg_{}.npy'.format(look))
keff_fiss1 = np.load('mydata/djinn_fission_1d/keff_small_norm_reg_{}.npy'.format(look))
phi_fiss2 = np.load('mydata/djinn_fission_1d/phi_small_norm_label_{}.npy'.format(look))
keff_fiss2 = np.load('mydata/djinn_fission_1d/keff_small_norm_label_{}.npy'.format(look))

# %% By Energy Level - small group
energy = display.gridPlot(np.load('mydata/uh3_20_enriched/energyGrid.npy'))
space = np.linspace(0,100,1000,endpoint=False)
en = 10
en = np.argmin(abs(energy-14e6))
# plt.plot(space,phi[:,en],'--',c='k',label='True, keff {}'.format(np.round(keff,5)))
# plt.plot(space,phi_fiss1[:,en],c='r',label='Fission, keff {}'.format(np.round(keff_fiss1,5)),alpha=0.6)
# plt.plot(space,phi_fiss2[:,en],c='b',label='Fission Labeled, keff {}'.format(np.round(keff_fiss2,5)),alpha=0.6)
# plt.ylabel('Flux'); 

plt.plot(space,display.error_calc(phi[:,en],phi_fiss1[:,en]),label='Fission',c='r',alpha=0.6)
plt.plot(space,display.error_calc(phi[:,en],phi_fiss2[:,en]),label='Fission Labeled',c='b',alpha=0.6)
plt.ylabel('Error (%)')

plt.grid(); plt.xlabel('Location (cm)')
plt.legend(loc='best') #,bbox_to_anchor=(1.5,0.5)
plt.title('Flux at Energy {:e} eV'.format(np.round(energy[en],4)))
plt.show()

# %% By Space - small group
x = 500
x = np.argmin(abs(space-80))
# plt.semilogx(energy,phi[x],'--',c='k',label='True, keff {}'.format(np.round(keff,5)))
# plt.semilogx(energy,phi_fiss1[x],c='r',label='Fission, keff {}'.format(np.round(keff_fiss1,5)),alpha=0.6)
# plt.semilogx(energy,phi_fiss2[x],c='b',label='Fission Labeled, keff {}'.format(np.round(keff_fiss2,5)),alpha=0.6)
# plt.ylabel('Flux'); 

plt.semilogx(energy,display.error_calc(phi[x],phi_fiss1[x]),label='Fission',c='r',alpha=0.6)
plt.semilogx(energy,display.error_calc(phi[x],phi_fiss2[x]),label='Fission Labeled',c='b',alpha=0.6)
plt.ylabel('Error (%)')

plt.grid(); plt.xlabel('Energy (eV)')
plt.legend(loc='best')
plt.title('Flux at Location {} cm'.format(space[x]))
plt.show()


# %% data collection - fission/scatter/both
import numpy as np
import glob
import discrete1.initialize as ex
import matplotlib.pyplot as plt
from discrete1.util import display
_,splits = ex.eigen_djinn.boundaries(0.05,symm=True)
# True Results
phi = np.sort(glob.glob('mydata/djinn_true_1d/phi_*'))
phi = [np.load(ii) for ii in phi]
keff = np.sort(glob.glob('mydata/djinn_true_1d/keff_*'))
keff = [np.load(ii) for ii in keff]

# Scatter DJINN - Labeled and Unlabeled
phi_scat = np.sort(glob.glob('mydata/djinn_scatter_1d/phi_None_reg*'))
phi_scat = [np.load(ii) for ii in phi_scat]
keff_scat = np.sort(glob.glob('mydata/djinn_scatter_1d/keff_None_reg*'))
keff_scat = [np.load(ii) for ii in keff_scat]

phi_scat_lab = np.sort(glob.glob('mydata/djinn_scatter_1d/phi_None_label*'))
phi_scat_lab = [np.load(ii) for ii in phi_scat_lab]
keff_scat_lab = np.sort(glob.glob('mydata/djinn_scatter_1d/keff_None_label*'))
keff_scat_lab = [np.load(ii) for ii in keff_scat_lab]

# Fission DJINN - Labeled and Unlabeled
phi_fis = np.sort(glob.glob('mydata/djinn_fission_1d/phi_None_reg*'))
phi_fis = [np.load(ii) for ii in phi_fis]
keff_fis = np.sort(glob.glob('mydata/djinn_fission_1d/keff_None_reg*'))
keff_fis = [np.load(ii) for ii in keff_fis]

phi_fis_lab = np.sort(glob.glob('mydata/djinn_fission_1d/phi_None_label*'))
phi_fis_lab = [np.load(ii) for ii in phi_fis_lab]
keff_fis_lab = np.sort(glob.glob('mydata/djinn_fission_1d/keff_None_label*'))
keff_fis_lab = [np.load(ii) for ii in keff_fis_lab]

# Both DJINN - Labeled and Unlabeled
phi_both = np.sort(glob.glob('mydata/djinn_both_1d/phi_None_reg*'))
phi_both = [np.load(ii) for ii in phi_both]
keff_both = np.sort(glob.glob('mydata/djinn_both_1d/keff_None_reg*'))
keff_both = [np.load(ii) for ii in keff_both]

phi_both_lab = np.sort(glob.glob('mydata/djinn_both_1d/phi_None_label*'))
labels = [ii.split('label_')[1].split('.')[0] for ii in phi_both_lab]
phi_both_lab = [np.load(ii) for ii in phi_both_lab]
keff_both_lab = np.sort(glob.glob('mydata/djinn_both_1d/keff_None_label*'))
keff_both_lab = [np.load(ii) for ii in keff_both_lab]


# %% By Energy Level
enrichment = 2 # 15%

energy = display.gridPlot(np.load('mydata/uh3_20_enriched/energyGrid.npy'))
space = np.linspace(0,100,1000,endpoint=False)
en = 10
# en = np.argmin(abs(energy-14e6))
for ii in [0,2,4]:
    plt.plot(space,phi[ii][:,en],'--',c='k',label='True, keff {}'.format(np.round(keff[enrichment],5)))
    
    # plt.plot(space,phi_scat[ii][:,en],c='r',label='Scatter, keff {}'.format(np.round(keff_scat[enrichment],3)),alpha=0.6)
    # plt.plot(space,phi_scat_lab[ii][:,en],c='b',label='Scatter Labeled, keff {}'.format(np.round(keff_scat_lab[enrichment],3)),alpha=0.6)
    
    plt.plot(space,phi_fis[ii][:,en],c='r',label='Fission, keff {}'.format(np.round(keff_fis[enrichment],5)),alpha=0.6)
    plt.plot(space,phi_fis_lab[ii][:,en],c='b',label='Fission Labeled, keff {}'.format(np.round(keff_fis_lab[enrichment],5)),alpha=0.6)
    
    # plt.plot(space,phi_both[ii][:,en],c='r',label='Both, keff {}'.format(np.round(keff_both[enrichment],3)),alpha=0.6)
    # plt.plot(space,phi_both_lab[ii][:,en],c='b',label='Both Labeled, keff {}'.format(np.round(keff_both_lab[enrichment],3)),alpha=0.6)

plt.grid(); plt.xlabel('Location (cm)')
plt.ylabel('Flux'); plt.legend(loc='best')
plt.title('Flux at Energy {:e} eV'.format(np.round(energy[en],4)))
plt.show()

# %% By Space
enrichment = 1 # 5%
x = 500
x = np.argmin(abs(space-90))
plt.plot(energy,phi[enrichment][x],'--',c='k',label='True, keff {}'.format(np.round(keff[enrichment],3)))

# plt.semilogx(energy,phi_scat[enrichment][x],c='r',label='Scatter, keff {}'.format(np.round(keff_scat[enrichment],3)),alpha=0.6)
# plt.semilogx(energy,phi_scat_lab[enrichment][x],c='b',label='Scatter Labeled, keff {}'.format(np.round(keff_scat_lab[enrichment],3)),alpha=0.6)

# plt.semilogx(energy,phi_fis[enrichment][x],c='r',label='Fission, keff {}'.format(np.round(keff_fis[enrichment],3)),alpha=0.6)
# plt.semilogx(energy,phi_fis_lab[enrichment][x],c='b',label='Fission Labeled, keff {}'.format(np.round(keff_fis_lab[enrichment],3)),alpha=0.6)

plt.semilogx(energy,phi_both[enrichment][x],c='r',label='Both, keff {}'.format(np.round(keff_both[enrichment],3)),alpha=0.6)
plt.semilogx(energy,phi_both_lab[enrichment][x],c='b',label='Both Labeled, keff {}'.format(np.round(keff_both_lab[enrichment],3)),alpha=0.6)

plt.grid(); plt.xlabel('Energy (eV)')
plt.ylabel('Flux'); plt.legend(loc='best')
plt.title('Flux at Location {} cm'.format(space[x]))
plt.show()



# %% Hang ups on DJINN Sn
import pstats
p = pstats.Stats('outputCProfile.txt')
p.sort_stats('cumulative')#.print_stats(40)

with open('CprofileDJINNSnFission.txt', 'w') as stream:
    stats = pstats.Stats('outputCProfile.txt', stream=stream)
    stats.sort_stats('cumulative').print_stats()

# %% Comparing first iteration
import numpy as np
import matplotlib.pyplot as plt
from discrete1.util import display

look = '15'
true_fission = np.load('mydata/djinn_true_1d/fission2_{}.npy'.format(look))[:,:550]
true_phi = true_fission[0,:,1:].copy(); true_djinn = true_fission[1,:,1:].copy()

lab_fission = np.load('mydata/djinn_fission_1d/fission_small_norm_label_{}.npy'.format(look))[:,:550]
lab_phi = lab_fission[0,:,1:].copy(); lab_djinn = lab_fission[1,:,1:].copy()

reg_fission = np.load('mydata/djinn_fission_1d/fission_small_norm_reg_{}.npy'.format(look))[:,:550]
reg_phi = reg_fission[0,:,1:].copy(); reg_djinn = reg_fission[1,:,1:].copy()

# %% Comparing Phis
energy = display.gridPlot(np.load('mydata/uh3_20_enriched/energyGrid.npy'))
space = np.linspace(45,100,550,endpoint=False)
space = np.arange(550)
en = 10
en = np.argmin(abs(energy-14e6))
plt.scatter(space,true_phi[:,en],c='k',label='Phi True')
plt.scatter(space,reg_phi[:,en],c='r',label='Phi Reg',alpha=0.6)
plt.scatter(space,lab_phi[:,en],c='b',label='Phi Label',alpha=0.6)
plt.grid(); plt.xlabel('Location (cm)')
plt.ylabel('Flux'); plt.legend(loc='best')
plt.title('Flux at Energy {:e} eV'.format(np.round(energy[en],4)))
plt.ylim([0,0.005])
plt.show()

x = 500
x = np.argmin(abs(space-80))
plt.semilogx(energy,true_phi[x],'--',c='k',label='Phi True')
plt.semilogx(energy,reg_phi[x],c='r',label='Phi Reg',alpha=0.6)
plt.semilogx(energy,lab_phi[x],c='b',label='Phi Label',alpha=0.6)
plt.grid(); plt.xlabel('Energy (eV)')
plt.ylabel('Flux'); plt.legend(loc='best')
plt.title('Flux at Location {} cm'.format(space[x]))
plt.show()

# %% Comparing DJINN
energy = display.gridPlot(np.load('mydata/uh3_20_enriched/energyGrid.npy'))
space = np.linspace(45,100,550,endpoint=False)
en = 10
en = np.argmin(abs(energy-14e6))
plt.scatter(space,true_djinn[:,en],c='k',label='Phi True')
plt.scatter(space,reg_djinn[:,en],c='r',label='Phi Reg',alpha=0.6)
plt.scatter(space,lab_djinn[:,en],c='b',label='Phi Label',alpha=0.6)
plt.grid(); plt.xlabel('Location (cm)')
plt.ylabel('Flux'); plt.legend(loc='best')
plt.title('Flux at Energy {:e} eV'.format(np.round(energy[en],4)))
plt.ylim([0,0.0005])
plt.show()

x = 500
x = np.argmin(abs(space-80))
plt.semilogx(energy,true_djinn[x],'--',c='k',label='Phi True')
plt.semilogx(energy,reg_djinn[x],c='r',label='Phi Reg',alpha=0.6)
plt.semilogx(energy,lab_djinn[x],c='b',label='Phi Label',alpha=0.6)
plt.grid(); plt.xlabel('Energy (eV)')
plt.ylabel('Flux'); plt.legend(loc='best')
plt.title('Flux at Location {} cm'.format(space[x]))
plt.show()

# %%
from djinn import djinn
from discrete1.util import nnets,sn
import discrete1.rogue as r
import matplotlib.pyplot as plt
import numpy as np

_,splits = r.eigen_djinn.boundaries(0.15,symm=True)
nums1 = nnets.djinn_metric('scatter_1d/norm/djinn_reg/error/model*',clean=True)
model1 = djinn.load(model_name='scatter_1d/norm/djinn_reg/model_{}'.format(nums1))

nums2 = nnets.djinn_metric('fission_1d/norm/djinn_reg/error/model*',clean=True)
model2 = djinn.load(model_name='fission_1d/norm/djinn_reg/model_{}'.format(nums2))

phi = sn.cat(np.load('mydata/djinn_true_1d/phi_15.npy'),splits['djinn'])
_,_,_,_,_,scatter,fission,_,_,_ = r.eigen_djinn.variables(0.15,symm=True)
scatter = sn.cat(scatter,splits['djinn'])
fission = sn.cat(fission,splits['djinn'])

# %%
phi_norm = phi.copy()/np.linalg.norm(phi.copy(),axis=1)[:,None]
# phi_norm = phi.copy()

djinn_output_scatter = model1.predict(phi_norm)
djinn_output_fission = model2.predict(phi_norm)

scale = np.sum(phi*np.sum(scatter,axis=1),axis=1)/np.sum(djinn_output_scatter,axis=1)

mat_mult_scatter = np.einsum('ijk,ik->ij',scatter,phi)
mat_mult_fission = np.einsum('ijk,ik->ij',fission,phi)

for ii in [0,9,19,39,59,69,79,86]:
    plt.plot(djinn_output_scatter[:,ii],label='DJINN',c='r',alpha=0.6)
    plt.plot(mat_mult_scatter[:,ii],label='True',c='k',ls='--')
    plt.plot((scale[:,None]*djinn_output_scatter)[:,ii],label='DJINN Scale',c='b',alpha=0.6)
    plt.legend(loc='best'); plt.grid()
    plt.title('Scatter at Energy Level {}'.format(ii+1)); plt.xlabel('Spatial Cell')
    plt.ylabel('Matrix Multiplication Magnitude'); plt.show()
    
    # plt.plot(djinn_output_fission[ii],label='DJINN')
    # plt.plot(mat_mult_fission[ii],label='True')
    # plt.legend(loc='best'); plt.grid()
    # plt.title('Fission {}'.format(ii)); plt.show()


    # def scale_scatter(self,phi,djinn_ns):
    #     import numpy as np
    #     from discrete1.util import sn
    #     if np.sum(phi) == 0:
    #         return np.zeros((sn.cat(phi,self.splits['djinn']).shape))
    #     interest = sn.cat(phi,self.splits['djinn'])
    #     scale = np.sum(interest*np.sum(sn.cat(self.scatter,self.splits['djinn']),axis=1),axis=1)/np.sum(djinn_ns,axis=1)
    #     return scale[:,None]*djinn_ns

    # def scale_fission(self,phi,djinn_ns):
    #     import numpy as np
    #     from discrete1.util import sn
    #     if np.sum(phi) == 0:
    #         return np.zeros((sn.cat(phi,self.splits['djinn']).shape))
    #     if self.dtype == 'scatter':
    #         return np.einsum('ijk,ik->ij',self.chiNuFission,phi) 
    #     interest = sn.cat(phi,self.splits['djinn'])
    #     scale = np.sum(interest*np.sum(sn.cat(self.chiNuFission,self.splits['djinn']),axis=1),axis=1)/np.sum(djinn_ns,axis=1)
    #     regular = np.einsum('ijk,ik->ij',sn.cat(self.chiNuFission,self.splits['keep']),sn.cat(phi,self.splits['keep']))
    #     return np.vstack((regular,scale[:,None]*djinn_ns))
