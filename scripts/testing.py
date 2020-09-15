import numpy as np
import matplotlib.pyplot as plt
import discrete1.setup as s
from discrete1.util import sn
import glob
# import pandas as pd
from djinn import djinn

# %%
scatter,fission = s.problem1.scatter_fission(0.15,'carbon')

phi1 = np.load('mydata/djinn_carbon/phi_15.npy')
phi2 = np.load('mydata/eNDe__true_1d/phi_phi_dummy1_15.npy')
phi3 = np.load('mydata/eNDe__true_1d/phi_smult_dummy1_15.npy')

keff1 = np.load('mydata/djinn_carbon/keff_15.npy')
keff2 = np.load('mydata/eNDe__true_1d/keff_phi_dummy1_15.npy')
keff3 = np.load('mydata/eNDe__true_1d/keff_smult_dummy1_15.npy')
plt.plot(sn.totalFissionRate(scatter,phi1),label='True, keff {}'.format(np.round(keff1,5)),c='k',ls='--')
plt.plot(sn.totalFissionRate(scatter,phi2),label='Dummy 1, keff {}'.format(np.round(keff2,5)),c='r',alpha=0.6)
plt.plot(sn.totalFissionRate(scatter,phi3),label='Dummy 2, keff {}'.format(np.round(keff3,5)),c='b',alpha=0.6)
plt.grid(); plt.legend(loc='best')
# plt.savefig('../Desktop/scipy_fixed.png',bbox_inches='tight')
plt.show()

plt.plot(sn.totalFissionRate(fission,phi1),label='True',c='k',ls='--')
plt.plot(sn.totalFissionRate(fission,phi2),label='Dummy 1',c='r',alpha=0.6)
plt.plot(sn.totalFissionRate(fission,phi3),label='Dummy 2',c='b',alpha=0.6)
plt.grid(); plt.legend(loc='best')
plt.show()

# np.load('mydata/eNDe__true_1d/keff_phi_dummy1_15.npy')

# %%
scatter,fission = s.problem1.scatter_fission(0.15,'carbon')

model = djinn.load('src_scatter_1d/carbon15/djinn_reg/model_005002')

ftype = 'both'
phi1 = np.load('mydata/ae_source_true_1d/phi_carbon_15.npy')
phi2 = model.predict(phi1)
# phi2 = np.load('mydata/djinn_src_carbon15/{}_phi_reg_15.npy'.format(ftype))
# phi3 = np.load('mydata/djinn_src_carbon15/{}_phi_label_15.npy'.format(ftype))

# %%

phi1 = np.load('mydata/ae_source_true_1d/phi_carbon_15.npy')
phi2 = np.load('mydata/eNDe__source_true_1d/phi_phi_dummy1_15.npy')

plt.semilogy(sn.totalFissionRate(scatter,phi1),label='True',c='k',ls='--')
plt.semilogy(sn.totalFissionRate(scatter,phi2),label='Dummy 1',c='r',alpha=0.6)
plt.grid(); plt.legend(loc='best')
plt.show()

plt.semilogy(sn.totalFissionRate(fission,phi1),label='True',c='k',ls='--')
plt.semilogy(sn.totalFissionRate(fission,phi2),label='Dummy 1',c='r',alpha=0.6)
plt.grid(); plt.legend(loc='best')
plt.show()



# %%

# ftype = 'fission'
for ftype in ['fission','scatter','both']:
    phi1 = np.load('mydata/ae_source_true_1d/phi_carbon_15.npy')
    phi2 = np.load('mydata/djinn_src_carbon15/{}_phi_reg_15.npy'.format(ftype))
    phi3 = np.load('mydata/djinn_src_carbon15/{}_phi_label_15.npy'.format(ftype))
    
    plt.semilogy(sn.totalFissionRate(scatter,phi1),label='True',ls='--',c='k')
    plt.semilogy(sn.totalFissionRate(scatter,phi2),label='Reg',c='r',alpha=0.6)
    plt.semilogy(sn.totalFissionRate(scatter,phi3),label='Label',c='b',alpha=0.6)
    # plt.semilogy(sn.totalFissionRate(scatter,phi4),label='Both',c='g',alpha=0.6)
    plt.grid(); plt.legend(loc='best')
    plt.title('DJINN {}'.format(ftype))
    plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from discrete1.util import sn
from discrete1.setup import problem1

scatter,fission = problem1.scatter_fission(0.15,'carbon')
# ftype = 'fission'

phi1 = np.load('mydata/ae_source_true_1d/phi_carbon_15.npy')
phi2 = np.load('mydata/djinn_src_carbon15/scatter_phi_reg_15.npy')
# phi3 = np.load('mydata/djinn_src_carbon15/{}_phi_label_15.npy'.format(ftype))

plt.semilogy(sn.totalFissionRate(scatter,phi1),label='True',ls='--',c='k')
plt.semilogy(sn.totalFissionRate(scatter,phi2),label='DJINN',c='r',alpha=0.6)
# %%
plt.semilogy(sn.totalFissionRate(scatter15,phi1),label='Old',ls='--',c='k')
plt.semilogy(sn.totalFissionRate(scatter15,phi2),label='New',ls='--',c='r')
plt.grid(); plt.legend(loc='best'); plt.show()

# %%
plt.semilogy(sn.totalFissionRate(fission15,phi1),label='Old',ls='--',c='k')
plt.semilogy(sn.totalFissionRate(fission15,phi2),label='New',ls='--',c='r')
plt.grid(); plt.legend(loc='best'); plt.show()

# %%
phi2 /= np.linalg.norm(phi2)
plt.semilogy(sn.totalFissionRate(scatter,phi1),label='True',ls='--',c='k')
plt.semilogy(sn.totalFissionRate(scatter,phi2),label='DJINN',c='r',alpha=0.6)



# %%
from discrete1.setup import func,problem1,ex_sources
import numpy as np
from discrete1.util import sn
import matplotlib.pyplot as plt
import scipy.optimize as op

# def func(x):
#     return Q + weight*psi_bottom - 0.5*total[cell]*psi_bottom - 0.5*x*total[cell])/weight

def source_iteration(Q,psi_bottom,weight,guess,cell):
    import numpy as np
    old = guess[None,:].copy()
    converged = 0; count = 1
    alpha_bottom = psi_bottom * total[ii]
    while not (converged):
        # alpha_top = 2*((Q + weight*psi_bottom - 0.5*alpha_bottom)/(weight*old))
        # new = alpha_top/total[cell]
        
        # new = 2*(mult + source + weight*psi_bottom - 0.5*alpha_bottom - weight*old)/self.total[cell]
        alpha_top = old*total[ii]
        # alpha_top[alpha_top < 0] = 0
        # if alpha_top > 1e3:
        #     print()
        new = (Q + weight*psi_bottom - 0.5*alpha_bottom - 0.5*alpha_top)/weight
        new[new < 0] = 1e-30
        new[np.isnan(new)] = 0
        change = np.mean(abs(new-old)/new)
        # print('Change',change,'count',count)
        converged = (change < 1e-10) or (count >= 100)
        # print(np.sum(old))
        old = new.copy(); count += 1
    # print(np.sum((mult + source + weight*psi_bottom - 0.5*alpha_bottom)-(weight*old+0.5*alpha_top)))
    return new # of size (1 x G_hat)

# %%

phi_old = func.initial_flux('carbon_source')
G,N,mu,w,total,scatter,fission,L,R,I = problem1.variables(0.15,'carbon')
inv_delta = float(I)/R

smult = np.einsum('ijk,ik->ij',scatter,phi_old)
fmult = np.einsum('ijk,ik->ij',fission,phi_old)
mult = smult + fmult

phi_true = np.zeros((I,G))
left_psis = np.zeros((N,I,G))
right_psis = np.zeros((N,I,G))
external = ex_sources.source1(I,G)
half_total = total*0.5
# Left to right
for n in range(N):
    weight = (mu[n]*inv_delta)
    psi_bottom = np.zeros((1,G))
    for ii in range(I):
        psi_top = (mult[ii] + external[ii] + psi_bottom * (weight-half_total[ii]))/(weight+half_total[ii])
        left_psis[n,ii] = psi_top.copy()
        phi_true[ii] = phi_true[ii] + (w[n] * func.diamond_diff(psi_top,psi_bottom))
        psi_bottom = psi_top.copy()
    # Reflective right to left
    for ii in range(I-1,-1,-1):
        psi_top = psi_bottom.copy()
        psi_bottom = (mult[ii] + external[ii] + psi_top * (weight-half_total[ii]))/(weight+half_total[ii])
        right_psis[n,ii] = psi_bottom.copy()
        phi_true[ii] = phi_true[ii] +  (w[n] * func.diamond_diff(psi_top,psi_bottom))


# %%
# psi_look = left_psis[0,0].copy()
# print(psi_look.shape)
# weight = (mu[0]*inv_delta)
# ans = (mult[0]+external[0])/(psi_look*(weight+0.5*total[ii]))

# ans2 = (mult[0]+external[0]-psi_look*0.5*total[ii])/weight

# plt.plot(psi_look,c='k',ls='--')
# plt.plot(ans2,alpha=0.6)

np.save('discrete1/data/left_psis',left_psis)
np.save('discrete1/data/right_psis',right_psis)
# %%
plt.semilogy(sn.totalFissionRate(scatter,phi_true),label='True',alpha=0.7)
plt.grid(); plt.legend(loc='best')

# %%
# guess = func.initial_flux('carbon_source')
guess = np.random.rand(I,G)
guess /= np.linalg.norm(guess)


def function(x,Q,total,weight):
    # print(((mult[ii]+external[ii] + weight*psi_bottom - 0.5*total[ii]*psi_bottom - 0.5*x*total[ii])/weight).shape)
    return (Q - 0.5*x*total)/weight
    # return (mult[ii]+external[ii] + weight*psi_bottom - 0.5*total[ii]*psi_bottom - 0.5*x*total[ii])/weight

phi = np.zeros((I,G))
for n in range(N):
    weight = (mu[n]*inv_delta)
    psi_bottom = np.zeros((G))
    # Left to right
    for ii in range(I):
        # psi_top = source_iteration(mult[ii]+external[ii],psi_bottom,weight,guess[ii],ii)
        # print(mult[ii].shape,weight.shape,psi_bottom.shape,external[ii].shape)
        # print(total[ii].shape)
        # print('Here')
        Q = mult[ii]+external[ii] + weight*psi_bottom - 0.5*total[ii]*psi_bottom
        xs = total[ii].copy()
        moo = weight.copy()
        psi_top = op.fixed_point(function,np.zeros((87)),args=(Q,xs,moo))
        # print('Here 2')
        # print(psi_top)
        psi_top[psi_top < 0] = 0
        phi[ii] = phi[ii] + (w[n] * func.diamond_diff(psi_top,psi_bottom))
        psi_bottom = psi_top.copy()
    for ii in range(I-1,-1,-1):
        psi_top = psi_bottom.copy()
        # psi_bottom = source_iteration(mult[ii]+external[ii],psi_top,weight,guess[ii],ii)
        Q = mult[ii]+external[ii] + weight*psi_top - 0.5*total[ii]*psi_top
        xs = total[ii].copy()
        moo = weight.copy()
        psi_bottom = op.fixed_point(function,np.zeros((87)),args=(Q,xs,moo))
        psi_bottom[psi_bottom < 0] = 0
        phi[ii] = phi[ii] +  (w[n] * func.diamond_diff(psi_top,psi_bottom))




# %%

plt.semilogy(sn.totalFissionRate(scatter,phi_true),label='True',alpha=0.7)
plt.semilogy(sn.totalFissionRate(scatter,phi),label='Source',alpha=0.7)
# plt.ylim([1,1e7])
plt.grid(); plt.legend(loc='best')


# %%
for gg in range(5):
    plt.plot(phi[:,gg],label='Source',alpha=0.6,c='r')
    plt.plot(phi_true[:,gg],label='True',c='k',ls='--')
    plt.grid(); plt.legend(loc='best')
    plt.title('Number {}'.format(gg))
    plt.show()




# %%
from discrete1.setup import func,problem1,ex_sources
import numpy as np
from discrete1.util import sn
import matplotlib.pyplot as plt
from djinn import djinn

model = 'src_scatter_1d/carbon15/djinn_reg/model_003002'
model = djinn.load(model)
scatter,fission = problem1.scatter_fission(0.15,'carbon')
data = np.load('mydata/model_data_djinn/carbon15_src_scatter_data.npy')
print(data.shape)

# %%

smult = data[1,:,1:].copy()
phi = data[0,:,1:].copy()

for ii in range(0,1296000,1000)[:5]:
    temp = model.predict(phi[ii:ii+1000])
    plt.plot(np.sum(temp,axis=1),label='DJINN',c='r',alpha=0.6)
    plt.plot(np.sum(smult[ii:ii+1000],axis=1),label='True',c='k',ls='--')
    plt.grid(); plt.legend(loc='best')
    plt.title('Iteration {}'.format(int(ii*0.001)))
    # plt.savefig('results_src_djinn/iteration_{}.png'.format(str(int(ii*0.001)).zfill(5)))
    plt.show()

# %%
from PIL import Image
import glob
frames = []
imgs = np.sort(glob.glob('results_src_djinn/*.png'))
for ii in imgs:
    temp = Image.open(ii)
    frames.append(temp)
    
frames[0].save('results_src_djinn/results.gif',format='GIF',
               append_images=frames[1:],save_all=True,duration=100,loop=0)


# %%
import glob

smult_add = np.sort(glob.glob('results_src_djinn/smult*'))
smult = [np.load(ii) for ii in smult_add]
fmult_add = np.sort(glob.glob('results_src_djinn/fmult*'))
fmult = [np.load(ii) for ii in fmult_add]

for ii in range(len(fmult)):
    # plt.plot(np.sum(fmult[ii],axis=1))
    plt.plot(np.sum(smult[ii],axis=1))
    plt.grid(); plt.show()



# %%
enrich = [0.05,0.12,0.15,0.25,0.27]
labels = ['05','12','15','25','27']
# enrich = [0.15]
# labels = ['15']
scatter = []; fission = []
for ii in enrich:    
    t1,t2 = s.problem1.scatter_fission(ii,'stainless')
    scatter.append(t1); fission.append(t2)
    del t1,t2

phi1 = np.sort(glob.glob('mydata/djinn_stainless/true_phi_[0-7][2-7]*'))
phi1 = [np.load(ii) for ii in phi1]
keff1 = np.sort(glob.glob('mydata/djinn_stainless/true_keff_[0-7][2-7]*'))
keff1 = [np.load(ii) for ii in keff1]
keff1 = sn.keff_correct(keff1)


phi2 = np.sort(glob.glob('mydata/djinn_stainless/fission_phi_label*'))
phi2 = [np.load(ii) for ii in phi2]
keff2 = np.sort(glob.glob('mydata/djinn_stainless/fission_keff_label*'))
keff2 = [np.load(ii) for ii in keff2]

phi3 = np.sort(glob.glob('mydata/djinn_stainless/scatter_phi_label*'))
phi3 = [np.load(ii) for ii in phi3]
keff3 = np.sort(glob.glob('mydata/djinn_stainless/scatter_keff_label*'))
keff3 = [np.load(ii) for ii in keff3]

# %%
for ii in range(len(enrich)):
    plt.plot(sn.totalFissionRate(fission[ii],phi1[ii]),label='True, keff {}'.format(np.round(keff1[ii],5)),c='k',ls='--')
    plt.plot(sn.totalFissionRate(fission[ii],phi2[ii]),label='Fission, keff {}'.format(np.round(keff2[ii],5)),c='r',alpha=0.6)
    plt.plot(sn.totalFissionRate(fission[ii],phi3[ii]),label='Scatter, keff {}'.format(np.round(keff3[ii],5)),c='b',alpha=0.6)
    plt.grid(); plt.legend(loc='best')
    plt.title('Fission Enrichment {}%'.format(labels[ii]))
    plt.show()
    
    # plt.plot(sn.totalFissionRate(scatter[ii],phi1[ii]),label='True, keff {}'.format(np.round(keff1[ii],5)),c='k',ls='--')
    # plt.plot(sn.totalFissionRate(scatter[ii],phi2[ii]),label='Fission, keff {}'.format(np.round(keff2[ii],5)),c='r',alpha=0.6)
    # plt.plot(sn.totalFissionRate(scatter[ii],phi3[ii]),label='Scatter, keff {}'.format(np.round(keff3[ii],5)),c='b',alpha=0.6)
    # plt.grid(); plt.legend(loc='best')
    # plt.title('Scatter Enrichment {}%'.format(labels[ii]))
    # plt.show()





# %%
import numpy as np
import glob
import matplotlib.pyplot as plt
from discrete1.setup import problem1
from discrete1.util import sn

enrich = '20'
add = np.sort(glob.glob('mydata/track_stainless_djinn/enrich_{}*'.format(enrich)))
scatter,fission = problem1.scatter_fission(int(enrich)*0.01,'stainless')
data = np.load(add[100])[0,:,1:]
real = np.load('mydata/djinn_stainless/true_phi_{}.npy'.format(enrich))
# print(add[1])

dim = np.linspace(0,data.shape[0],int(data.shape[0]/1000),endpoint=False).astype('int')
for ii in dim:
    plt.plot(sn.totalFissionRate(fission,real),label='True',c='k',ls='--')
    temp = data[ii:ii+1000].copy()
    temp /= np.linalg.norm(temp)
    # plt.plot(sn.totalFissionRate(fission,data[ii:ii+1000]),label='Iteration',c='r',alpha=0.6)
    plt.plot(sn.totalFissionRate(fission,temp),label='Iteration',c='r',alpha=0.6)
    plt.grid(); plt.title('Iteration {}'.format(int(ii*0.001)))
    plt.legend(loc='best'); plt.show()



