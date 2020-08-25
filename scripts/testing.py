import numpy as np
import matplotlib.pyplot as plt
import discrete1.setup as s
from discrete1.util import sn
import glob
# import pandas as pd
# from djinn import djinn
# %%
_,_,_,_,_,scatter15,fission15,_,_,_ = s.problem.variables(0.15,'carbon','carbon_full')

phi1 = np.load('mydata/ae_source_true_1d/phi_carbon_15.npy') # No Normalization

# %%

plt.semilogy(sn.totalFissionRate(scatter15,phi1),label='Old',ls='--',c='k')
plt.grid(); plt.legend(loc='best')

# %%
phi2 = np.load('mydata/ae_source_true_1d/phi_carbon_15.npy') # Normalization


# %%
plt.semilogy(sn.totalFissionRate(scatter15,phi1),label='Old',ls='--',c='k')
plt.semilogy(sn.totalFissionRate(scatter15,phi2),label='New',ls='--',c='r')
plt.grid(); plt.legend(loc='best'); plt.show()

# %%
plt.semilogy(sn.totalFissionRate(fission15,phi1),label='Old',ls='--',c='k')
plt.semilogy(sn.totalFissionRate(fission15,phi2),label='New',ls='--',c='r')
plt.grid(); plt.legend(loc='best'); plt.show()