import numpy as np
import matplotlib.pyplot as plt
from discrete1.setup import problem1,func
from discrete1.util import sn,nnets

# %%

def source_iteration(sources,total_xs,psi_bottom,weight,guess,normalize=False):
    import numpy as np
    from discrete1.util import nnets

    old = guess[None,:].copy()
    alpha_bottom = 0.5*psi_bottom*total_xs
    if normalize:
        # alpha_bottom,a1maxi,a1mini = nnets.normalize(alpha_bottom)
        # alpha_bottom = nnets.phi_normalize_single(alpha_bottom,normalize[0],normalize[1])
        alpha_bottom = alpha_bottom * normalize[0]/normalize[1]
    converged = 0; count = 1
    new = np.zeros((1,G))
    while not (converged):
        alpha_top = 0.5*total_xs*old
        if normalize:
            # alpha_top = nnets.phi_normalize_single(alpha_top,normalize[0],normalize[1])
            alpha_top = alpha_top * normalize[0]/normalize[1]
            
            # beta_top = weight*psi_bottom
            # beta_top = nnets.phi_normalize_single(weight*psi_bottom,normalize[0],normalize[1])
            beta_top = (weight*psi_bottom)* normalize[0]/normalize[1]
            
            # denominator = old*weight+alpha_top
            # denominator = nnets.phi_normalize_single(old*weight,normalize[0],normalize[1])+alpha_top
            denominator = (old*weight) * normalize[0]/normalize[1] + alpha_top
        else:
            beta_top = weight*psi_bottom
            denominator = (old*weight+alpha_top)
        if np.argwhere(denominator == 0).shape[0] > 0:
            ind = np.argwhere(denominator != 0)[:,1].flatten()
            new = np.zeros((old.shape))
            new[:,ind] = (old * (sources + beta_top - alpha_bottom))[:,ind]/denominator[:,ind]
        else:    
            new = old * ((sources + beta_top - alpha_bottom)/denominator)
        new[np.isnan(new)] = 0; #new[np.isinf(new)] = 10
        change = np.argwhere(abs(old-new) < 1e-12)
        converged = (len(change) == G) or (count >= 5000)
        old = new.copy(); count += 1

    return new 

# %%
# Set up the original problem

G,N,mu,w,total,scatter,fission,L,R,I = problem1.variables(0.15,'carbon')
inv_delta = float(I)/R
phi_old = func.initial_flux('carbon')

smult = np.einsum('ijk,ik->ij',scatter,phi_old)
source = np.einsum('ijk,ik->ij',fission,phi_old)
mult = smult + source

# %%

phi_sweep = np.zeros((I,G))
phi_source = np.zeros((I,G))

angular_sweep = np.zeros((N,I,G))
angular_source = np.zeros((N,I,G))

guess = phi_old.copy()
for n in range(N):
    weight = mu[n]*inv_delta
    psi_bottom_source = np.zeros((1,G))
    psi_bottom_sweep = np.zeros((1,G))
    for ii in range(I):
        psi_top_source = source_iteration(mult[ii],total[ii],psi_bottom_source,weight,guess[ii])
        phi_source[ii] = phi_source[ii] +  (w[n]* 0.5*(psi_top_source + psi_bottom_source))
        angular_source[n,ii] = 0.5*(psi_top_source + psi_bottom_source)
        # psi_top_source[psi_top_source < 0] = 0
        psi_bottom_source = psi_top_source.copy()
        
        psi_top_sweep = (mult[ii] + psi_bottom_sweep * (weight - 0.5*total[ii]))/(weight+0.5*total[ii])
        phi_sweep[ii] = phi_sweep[ii] +  (w[n]* 0.5*(psi_top_sweep + psi_bottom_sweep))
        angular_sweep[n,ii] = 0.5*(psi_top_sweep + psi_bottom_sweep)
        psi_bottom_sweep = psi_top_sweep.copy()
        
    for ii in range(I-1,-1,-1):
        psi_top_source = psi_bottom_source.copy()
        psi_bottom_source = source_iteration(mult[ii],total[ii],psi_top_source,weight,guess[ii])
        phi_source[ii] = phi_source[ii] +  (w[n]* 0.5*(psi_top_source + psi_bottom_source))
        angular_source[n,ii] = 0.5*(psi_top_source + psi_bottom_source)
        # psi_top_source[psi_top_source < 0] = 0
        
        psi_top_sweep = psi_bottom_sweep.copy()
        psi_bottom_sweep = (mult[ii] + psi_top_sweep * (weight - 0.5*total[ii]))/(weight+0.5*total[ii])
        phi_sweep[ii] = phi_sweep[ii] +  (w[n]* 0.5*(psi_top_sweep + psi_bottom_sweep))
        angular_sweep[n,ii] = 0.5*(psi_top_sweep + psi_bottom_sweep)
    print('Angle {}\n==============='.format(n))
    print('Source {}'.format(np.sum(phi_source)))
    print('Sweep {}\n'.format(np.sum(phi_sweep)))
        
# %%

for n in range(N):
    plt.plot(np.sum(angular_source[n],axis=1), label='Source',c='r',alpha=0.6)
    plt.plot(np.sum(angular_sweep[n],axis=1),label='Sweep',c='b',alpha=0.6)
    plt.grid(); plt.legend(loc='best'); 
    plt.title('Angle {}'.format(n)); plt.show()
    
    
# %% Normalize the data

# def z_score()

G,N,mu,w,total,scatter,fission,L,R,I = problem1.variables(0.15,'carbon')
inv_delta = float(I)/R
phi_old = func.initial_flux('carbon')
smult = np.einsum('ijk,ik->ij',scatter,phi_old)
source = np.einsum('ijk,ik->ij',fission,phi_old)

# pmean = np.mean(phi_old,axis=1); psigma = np.std(phi_old,axis=1)
# phi_old = (phi_old-pmean[:,None])/psigma[:,None]

pmaxi = np.max(phi_old,axis=1); pmini = np.min(phi_old,axis=1)
smaxi = np.max(smult,axis=1); smini = np.min(smult,axis=1)
fmaxi = np.max(source,axis=1); fmini = np.min(source,axis=1)
# phi_old,pmaxi,pmini = nnets.normalize(phi_old,verbose=True)
phi_old = phi_old * (pmaxi/pmini)[:,None]

smult = np.einsum('ijk,ik->ij',scatter,phi_old)
source = np.einsum('ijk,ik->ij',fission,phi_old)

# print(np.sum(smult))
# smult,smaxi,smini = nnets.normalize(smult,verbose=True)
# smult = nnets.phi_normalize(smult,pmaxi,pmini)
# smult = smult * (smaxi/smini)[:,None]
# print(np.sum(smult))

# source,fmaxi,fmini = nnets.normalize(source,verbose=True)
# print(np.sum(source))
# source = nnets.phi_normalize(source,pmaxi,pmini)
# source = source * (fmaxi/fmini)[:,None]
source[np.isnan(source)] = 0
# print(np.sum(source))
mult = smult + source


# %%
phi_old = func.initial_flux('carbon')
smult = np.einsum('ijk,ik->ij',scatter,phi_old)
source = np.einsum('ijk,ik->ij',fission,phi_old)

pmaxi = np.max(phi_old,axis=1); pmini = np.min(phi_old,axis=1)
smaxi = np.max(smult,axis=1); smini = np.min(smult,axis=1)
fmaxi = np.max(source,axis=1); fmini = np.min(source,axis=1)


# phi_old2 = nnets.normalize(phi_old)
phi_old2 = phi_old * (pmaxi/pmini)[:,None]
smult2 = np.einsum('ijk,ik->ij',scatter,phi_old2)
source2 = np.einsum('ijk,ik->ij',fission,phi_old2)

# smult2 = nnets.phi_normalize(smult,pmaxi,pmini)
# source2 = nnets.phi_normalize(source,pmaxi,pmini)

# plt.semilogy(np.sum(phi_old,axis=1),label='True Phi')
# plt.plot(np.sum(phi_old2,axis=1),label='Normalize Phi')
# plt.semilogy(np.sum(smult,axis=1),label='True Scatter')
# plt.plot(np.sum(smult2,axis=1),label='Normalize Scatter')
plt.semilogy(np.sum(source,axis=1),label='True Fission')
plt.plot(np.sum(source2,axis=1),label='Normalize Fission')
plt.legend(loc='best'); plt.grid()

# %%

nphi_sweep = np.zeros((I,G))
nphi_source = np.zeros((I,G))

nangular_sweep = np.zeros((N,I,G))
nangular_source = np.zeros((N,I,G))


# from scipy.optimize import fixed_point

# def function(angle,top,total,weight,maxi,mini):
#     return top - (nnets.phi_normalize_single(0.5*total*angle,maxi,mini) + weight*angle)

guess = phi_old.copy()
for n in range(N):
    weight = mu[n]*inv_delta
    psi_bottom_source = np.zeros((1,G))
    psi_bottom_sweep = np.zeros((1,G))
    for ii in range(I):
        psi_top_source = source_iteration(mult[ii],total[ii],psi_bottom_source,weight,guess[ii],normalize=[pmaxi[ii],pmini[ii]])
        psi_top_source[psi_top_source < 0] = 0
        # psi_top_source = source_iteration(mult[ii],total[ii],psi_bottom_source,weight,guess[ii],normalize=pnorm[ii])
        # psi_top_source *= pnorm[ii]
        # nphi_source[ii] = nphi_source[ii] +  (w[n] * 0.5 * (psi_top_source + psi_bottom_source))
        # nphi_source[ii] = nphi_source[ii] +  (w[n]* 0.5*(nnets.phi_normalize_single(psi_top_source,pmaxi[ii],pmini[ii]) + nnets.phi_normalize_single(psi_bottom_source,pmaxi[ii],pmini[ii])))
        nphi_source[ii] = nphi_source[ii] +  (w[n]* 0.5 * ((psi_top_source + psi_bottom_source) * pmaxi[ii]/pmini[ii]))
        nangular_source[n,ii] = 0.5*(psi_top_source + psi_bottom_source)
        
        psi_bottom_source = psi_top_source.copy()
        
        # psi_top_sweep = (mult[ii] + weight*psi_bottom_sweep - nnets.normalize(psi_bottom_sweep* 0.5*total[ii]))
        # psi_top_sweep = nnets.normalize(psi_top_sweep)
        # psi_top_sweep = fixed_point(function,np.zeros((1,87)),args=(psi_top_sweep,total[ii],weight,pmaxi[ii],pmini[ii]),xtol=1,maxiter=1000,method='iteration')
        
        # nphi_sweep[ii] = nphi_sweep[ii] +  (w[n]* 0.5*(psi_top_sweep + psi_bottom_sweep))
        # nangular_sweep[n,ii] = 0.5*(psi_top_sweep + psi_bottom_sweep)
        # psi_bottom_sweep = psi_top_sweep.copy()
        
    for ii in range(I-1,-1,-1):
        psi_top_source = psi_bottom_source.copy()
        psi_bottom_source = source_iteration(mult[ii],total[ii],psi_top_source,weight,guess[ii],normalize=[pmaxi[ii],pmini[ii]])
        psi_bottom_source[psi_bottom_source < 0] = 0
        # psi_bottom_source = source_iteration(mult[ii],total[ii],psi_top_source,weight,guess[ii],normalize=pnorm[ii])
        # psi_bottom_source *= pnorm[ii]
        # nphi_source[ii] = nphi_source[ii] +  (w[n]* 0.5*(nnets.phi_normalize_single(psi_top_source,pmaxi[ii],pmini[ii]) + nnets.phi_normalize_single(psi_bottom_source,pmaxi[ii],pmini[ii])))
        nphi_source[ii] = nphi_source[ii] +  (w[n]* 0.5*((psi_top_source + psi_bottom_source) * pmaxi[ii]/pmini[ii]))# pmaxi[ii]/pmini[ii]
        nangular_source[n,ii] = 0.5*(psi_top_source + psi_bottom_source)
        
        
        # psi_top_sweep = psi_bottom_sweep.copy()
        # psi_bottom_sweep = (mult[ii] + weight*psi_top_sweep - nnets.normalize(psi_top_sweep* 0.5*total[ii]))
        # psi_bottom_sweep = fixed_point(function,np.zeros((1,87)),args=(psi_bottom_sweep,total[ii],weight),xtol=1e-3,maxiter=5000)
        # psi_bottom_sweep = nnets.normalize(psi_bottom_sweep)
        
        # nphi_sweep[ii] = nphi_sweep[ii] +  (w[n]* 0.5*(psi_top_sweep + psi_bottom_sweep))
        # nangular_sweep[n,ii] = 0.5*(psi_top_sweep + psi_bottom_sweep)
    print('Angle {}\n==============='.format(n))
    print('Source {}'.format(np.sum(nphi_source)))
    # print('Sweep {}\n'.format(np.sum(nphi_sweep)))
    
# %%
# nphi_source2 = nnets.unnormalize(nphi_source,pmaxi,pmini)
# nphi_source2 = nphi_source * (pmini/pmaxi)[:,None] * (smini/smaxi)[:,None] * (fmini/fmaxi)[:,None]
nphi_source2 = nphi_source * ((pmini)/(pmaxi))[:,None] 
print(np.sum(nphi_source2))

# plt.semilogy(np.sum(nphi_source2,axis=1),label='Norm',c='r',alpha=0.6)
# plt.plot(np.sum(phi_sweep,axis=1),label='True',c='k',ls='--')
# plt.grid(); plt.legend(loc='best'); plt.show()

# plt.plot(np.sum(nphi_source2,axis=1),label='Norm',c='r',alpha=0.6)
# plt.plot(np.sum(phi_sweep,axis=1),label='True',c='k',ls='--')
# plt.grid(); plt.legend(loc='best'); plt.show()

plt.semilogy(sn.totalFissionRate(scatter,nphi_source2),label='Norm',c='r',alpha=0.6)
plt.plot(sn.totalFissionRate(scatter,phi_sweep),label='True',c='k',ls='--')
plt.grid(); plt.legend(loc='best'); plt.show()

plt.semilogy(sn.totalFissionRate(fission,nphi_source2),label='Norm',c='r',alpha=0.6)
plt.plot(sn.totalFissionRate(fission,phi_sweep),label='True',c='k',ls='--')
plt.grid(); plt.legend(loc='best'); plt.show()


# %%

for n in range(N):
    plt.plot(np.sum(nangular_source[n],axis=1), label='Source',c='r',alpha=0.6)
    plt.plot(np.sum(angular_sweep[n],axis=1),label='Sweep',c='b',alpha=0.6)
    plt.grid(); plt.legend(loc='best'); 
    plt.title('Angle {}'.format(n)); plt.show()

# %%
pnorm2 = np.linalg.norm(nphi_source,axis=1)

plt.semilogy(np.sum(nphi_source/pnorm[:,None]*pnorm2[:,None],axis=1),label='Norm',c='r',alpha=0.6)
plt.plot(np.sum(phi_sweep,axis=1),label='True',c='k',ls='--')
plt.grid(); plt.legend(loc='best'); plt.show()

plt.plot(np.sum(nphi_source/pnorm[:,None]*pnorm2[:,None],axis=1),label='Norm',c='r',alpha=0.6)
plt.plot(np.sum(phi_sweep,axis=1),label='True',c='k',ls='--')
plt.grid(); plt.legend(loc='best'); plt.show()


# %%

tt = nnets.phi_normalize(phi_sweep,pmaxi,pmini)
tt.shape


