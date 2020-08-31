import numpy as np
import matplotlib.pyplot as plt
import discrete1.setup as s
from discrete1.util import sn
import glob
# import pandas as pd
from djinn import djinn

# %%
scatter,fission = s.problem1.scatter_fission(0.15,'carbon')

phi1 = np.load('mydata/ae_source_true_1d/phi_carbon_15.npy')
phi2 = np.load('mydata/eNDe__source_true_1d/phi_phi_dummy1_15.npy')
plt.semilogy(sn.totalFissionRate(scatter,phi1),label='True',c='r',alpha=0.6)
plt.semilogy(sn.totalFissionRate(scatter,phi2),label='Est.',c='b',alpha=0.6)
plt.grid(); plt.legend(loc='best')
plt.show()


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
ftype = 'fission'

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
np.random.seed(47)

phi = np.random.rand(1000,87)
norm = np.linalg.norm(phi)
nphi = phi/norm
scatter = np.random.rand(1000,87,87)
smult = np.einsum('ijk,ik->ij',scatter,phi)
nsmult = np.einsum('ijk,ik->ij',scatter,nphi)

phi_diff = abs(phi-nphi*norm)
print('Phi')
print('Sum',np.sum(phi_diff),'Max',np.max(phi_diff))

mult_diff = abs(smult-nsmult*norm)
print('Smult')
print('Sum',np.sum(mult_diff),'Max',np.max(mult_diff))

plt.plot(sn.totalFissionRate(scatter,phi)-sn.totalFissionRate(scatter,nphi*norm),label='Real',ls='--',c='k')
# plt.plot(,label='Norm',c='r',alpha=0.6)
plt.grid(); plt.legend(loc='best')

# %%
import numpy as np
dim = 3
# mult = np.ones((1,dim))
mult = np.random.rand(1,dim)
# source = np.ones((1,dim))*0.5
source = np.random.rand(1,dim)
# total = np.ones((1,dim))*2
total = np.random.rand(1,dim)
# psi_bottom = np.zeros((1,dim))
psi_bottom = np.random.rand(1,dim)
weight = 0.99

np.random.seed(47)
guess = np.random.rand(1,dim)


def source_iteration(mult,source,total,psi_bottom,weight,guess):
    import numpy as np
    # old = np.random.rand(1,self.gprime) # initial guess
    old = guess.copy()
    converged = 0; count = 1
    while not (converged):
        alpha_top = total*old        
        alpha_bottom = total*psi_bottom        
        new = (mult + source + weight*psi_bottom + 0.5*alpha_bottom)*old/(weight*old-0.5*alpha_top)
        # new[np.isnan(new)] = 0
        print('count {}'.format(count))
        print('Old',old,'\nNew',new)
        
        change = np.linalg.norm((new-old)/new)
        # print(np.isnan(old).sum())
        print('Change',change,'count',count)
        print('=========================')
        converged = (change < 1e-8) or (count >= 10)
        old = new.copy(); count += 1
    return new # of size (1 x G_hat)

ans = source_iteration(mult,source,total,psi_bottom,weight,guess)
print(ans)


# %%
old = ans.copy()
alpha_top = total*old
alpha_bottom = total*psi_bottom        
new = (mult + source + weight*psi_bottom + 0.5*alpha_bottom)*old/(weight*old-0.5*alpha_top)
print(old-new)





# %%
from discrete1.setup import func,problem1,ex_sources

phi_old = func.initial_flux('carbon_source')
G,N,mu,w,total,scatter,fission,L,R,I = problem1.variables(0.15,'carbon')
inv_delta = float(I)/R
n = 0
weight = (mu[n]*inv_delta)

smult = np.einsum('ijk,ik->ij',scatter,phi_old)
fmult = np.einsum('ijk,ik->ij',fission,phi_old)
mult = smult + fmult
psi_bottom = np.zeros((1,G))
phi = np.zeros((I,G))
all_psis = np.zeros((I,G))
external = ex_sources.source1(I,G)
half_total = total*0.5
# Left to right
for ii in range(I):
    psi_top = (mult[ii] + external[ii] + psi_bottom * (weight-half_total[ii]))/(weight+half_total[ii])
    all_psis[ii] = psi_top.copy()
    phi[ii] = phi[ii] + (w[n] * func.diamond_diff(psi_top,psi_bottom))
    psi_bottom = psi_top
# Reflective right to left
# for ii in range(self.I-1,-1,-1):
#     psi_top = psi_bottom
#     psi_bottom = (smult[ii] + external[ii] + psi_top * (weight-half_total[ii]))/(weight+half_total[ii])
#     phi[ii] = phi[ii] +  (self.w[n] * func.diamond_diff(psi_top,psi_bottom))

# %%

def source_iteration(mult,source,psi_bottom,weight,total):
    import numpy as np
    # old = guess[None,:].copy()
    old = np.random.rand(1,87)
    # old = source.copy()+0.0001
    converged = 0; count = 1
    
    while not (converged):
        alpha_top = 0.5*old*total
        alpha_bottom = psi_bottom*total
    # new = np.zeros((87))
    # for g in range(G):
    #     old = np.random.rand(1)
    #     while not (converged):
    #         alpha_top = old*total[g]
    #         alpha_bottom = psi_bottom[g]*total[g]      
    #         # print(alpha_top.shape)
    #         # print(alpha_bottom.shape)
    #         # print(mult.shape)
    #         # print(source.shape)
    #         # print(psi_bottom.shape)
    #         # print('Old',old.shape)
    #         new[g] = ((mult[g] + source[g] + weight*psi_bottom[g] + 0.5*alpha_bottom)+(-0.5*alpha_top))/weight
            
        # print(np.sum(alpha_bottom))
        alpha_top = ((mult + source + weight*psi_bottom + 0.5*alpha_bottom))/(weight*old)
        new = 2*alpha_top/total
        new[np.isnan(new)] = 0
        change = np.linalg.norm((new-old)/new)

        converged = (change < 1e-8) or (count >= 100)
        old = new.copy(); count += 1
        # count = 1
        # converged = 0
    # print(count)
    return new 

# psi1 = source_iteration(mult[1],external[1],all_psis[0],weight,all_psis[1],total[1])
phi1 = np.zeros((I,G))
psi_bottom = np.zeros((G))
for ii in range(I):
    psi_top = source_iteration(mult[ii],external[ii],psi_bottom,weight,total[ii])
    # print('Made it {}'.format(ii))
    phi1[ii] = phi[ii] + (w[n] * func.diamond_diff(psi_top,psi_bottom))
    psi_bottom = psi_top.copy()


# %%
# plt.semilogy(abs(psi1.flatten()),alpha=0.6); plt.plot(all_psis[1],alpha=0.6)
plt.plot(sn.totalFissionRate(scatter,phi),label='True',alpha=0.6)
plt.plot(sn.totalFissionRate(scatter,phi1),label='source',alpha=0.6)
# plt.ylim([0,30])
plt.grid(); plt.legend(loc='best')
plt.show()




# %%

def source_iteration(mult,source,psi_bottom,weight,guess,total):
    import numpy as np
    old = guess[None,:].copy()
    # old = np.random.rand(1,87)
    # old = source.copy()+0.0001
    converged = 0; count = 1
    while not (converged):
        alpha_top = old*total
        alpha_bottom = psi_bottom*total
        # for g in range(G):
        #     new[:,g] = (mult[:,g] + source[:,g] + weight*psi_bottom[:,g] + 0.5*alpha_bottom[:,g])/(weight*old[:,g]-0.5*alpha_top[:,g])
        # print(np.sum(alpha_bottom))
        new = ((weight*old-0.5*alpha_top)+(mult + source + weight*psi_bottom + 0.5*alpha_bottom))/old
        
        new[np.isnan(new)] = 0
        change = np.linalg.norm((new-old)/new)

        converged = (change < 1e-8) or (count >= 100)
        old = new.copy(); count += 1
    print(count)
    return new 






