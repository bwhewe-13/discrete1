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
plt.plot(sn.totalFissionRate(scatter,phi1),label='True',c='r',alpha=0.6)
plt.plot(sn.totalFissionRate(scatter,phi2),label='Est.',c='b',alpha=0.6)
plt.grid(); plt.legend(loc='best')
plt.show()

plt.plot(sn.totalFissionRate(fission,phi1),label='True',c='r',alpha=0.6)
plt.plot(sn.totalFissionRate(fission,phi2),label='Est.',c='b',alpha=0.6)
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

plt.semilogy(sn.totalFissionRate(scatter,phi1),label='True',ls='--',c='k')
plt.semilogy(np.sum(phi2,axis=1),label='Model',c='r',alpha=0.6)
# plt.semilogy(sn.totalFissionRate(scatter,phi2),label='Reg',c='r',alpha=0.6)
# plt.semilogy(sn.totalFissionRate(scatter,phi3),label='Label',c='b',alpha=0.6)


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
plt.semilogy(sn.totalFissionRate(scatter,phi_true),label='True',alpha=0.7)
plt.grid(); plt.legend(loc='best')

# %%
# guess = func.initial_flux('carbon_source')
guess = np.random.rand(I,G)
guess /= np.linalg.norm(guess)

phi = np.zeros((I,G))
for n in range(N):
    weight = (mu[n]*inv_delta)
    psi_bottom = np.zeros((1,G))
    # Left to right
    for ii in range(I):
        psi_top = source_iteration(mult[ii]+external[ii],psi_bottom,weight,guess[ii],ii)
        psi_top[psi_top < 0] = 0
        phi[ii] = phi[ii] + (w[n] * func.diamond_diff(psi_top,psi_bottom))
        psi_bottom = psi_top.copy()
    for ii in range(I-1,-1,-1):
        psi_top = psi_bottom.copy()
        psi_bottom = source_iteration(mult[ii]+external[ii],psi_top,weight,guess[ii],ii)
        psi_bottom[psi_bottom < 0] = 0
        phi[ii] = phi[ii] +  (w[n] * func.diamond_diff(psi_top,psi_bottom))


# # %%
# guess = func.initial_flux('carbon_source')
# guess /= np.linalg.norm(guess)

# # for ii in range(15):
# #     print(ii,'\n==========================================')
# #     psi_top = source_iteration(mult[ii]+external[ii],np.zeros((1,G)),weight,guess[ii],ii)
# #     print('\n==========================================')

# left2_psis = np.zeros((N,I,G))
# right2_psis = np.zeros((N,I,G))

# for n in range(N):
#     weight = (mu[n]*inv_delta)
#     psi_bottom = np.zeros((1,G))+0.0001
#     # Left to right
#     for ii in range(I):
#         # print(n,ii,'\n==========================================')
#         psi_top = source_iteration(mult[ii]+external[ii],psi_bottom,weight,guess[ii],ii)
#         psi_top[psi_top < 0] = 0
#         left2_psis[n,ii] = psi_top.copy()
        
#         # print('\n==========================================')
#         phi[ii] = phi[ii] + (w[n] * func.diamond_diff(psi_top,psi_bottom))
#         psi_bottom = psi_top.copy()
#     for ii in range(I-1,-1,-1):
#         psi_top = psi_bottom.copy()
#         # print(n,ii,'\n==========================================')
#         psi_bottom = source_iteration(mult[ii]+external[ii],psi_top,weight,guess[ii],ii)
#         psi_bottom[psi_bottom < 0] = 0
#         right2_psis[n,ii] = psi_bottom.copy()
        
#         # print('\n==========================================')
#         phi[ii] = phi[ii] +  (w[n] * func.diamond_diff(psi_top,psi_bottom))

# # %%

# for nn in range(N):
#     plt.imshow(left2_psis[nn],aspect='auto')
#     plt.colorbar()
#     plt.title('Left Angle {}'.format(nn))
#     plt.show()

# for nn in range(N):
#     plt.imshow(right2_psis[nn],aspect='auto')
#     plt.colorbar()
#     plt.title('Right Angle {}'.format(nn))
#     plt.show()

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




