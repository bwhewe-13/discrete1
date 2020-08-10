import numpy as np
import matplotlib.pyplot as plt
from discrete1.util import display,sn
import discrete1.rogue as r

# %%
energy = display.gridPlot()
phi15 = np.load('mydata/djinn_infinite/true/phi_27.npy').flatten()
keff15 = np.load('mydata/djinn_infinite/true/keff_27.npy')
phi15_1 = np.load('mydata/djinn_infinite/both/phi_plain_27.npy').flatten()
keff15_1 = np.load('mydata/djinn_infinite/both/keff_plain_27.npy')
phi15_2 = np.load('mydata/djinn_infinite/both/phi_enrich27.npy').flatten()
keff15_2 = np.load('mydata/djinn_infinite/both/keff_enrich27.npy')

fig,ax1 = plt.subplots(figsize=(10,8))
ln1 = ax1.semilogx(energy,phi15,c='k',ls='--',label='Original, keff {}'.format(np.round(keff15,5)),lw=2)
ln2 = ax1.semilogx(energy,phi15_1,c='r',alpha=0.5,label='DJINN, keff {}'.format(np.round(keff15_1,5)),lw=2)
ln3 = ax1.semilogx(energy,phi15_2,c='b',alpha=0.5,label='DJINN Label, keff {}'.format(np.round(keff15_2,5)),lw=2)
plt.grid(color='gray',linestyle='--'); ax1.set_xlabel('Energy (eV)',fontsize=16)
ax1.set_ylabel(r'Flux $\phi$',fontsize=16); #plt.legend(loc='upper center',prop={'size':14})
plt.title(r'UH$_3$ 27% Enrichment',fontsize=18); ax1.tick_params(axis='both',labelsize=12)

ax2 = ax1.twinx()
ln4 = ax2.semilogx(energy,display.error_calc(phi15,phi15_1),label='DJINN Error',ls='--',c='r',alpha=0.6,lw=2)
ln5 = ax2.semilogx(energy,display.error_calc(phi15,phi15_2),label='DJINN Label Error',ls='--',c='b',alpha=0.6,lw=2)

lns = ln1 + ln2 + ln4 + ln3 + ln5
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper center',prop={'size':14})

ax2.set_ylabel('Error Percentage',fontsize=16,color='r')
ax2.tick_params(axis='y',labelsize=12,labelcolor='r')
# plt.savefig('paper_graphs/infinite27.png',bbox_inches='tight')
plt.show()

# %%
energy = display.gridPlot()
high_ind = np.argmin(abs(energy-1e5))
low_ind = np.argmin(abs(energy-1))+1
space = np.linspace(0,100,1000,endpoint=False)

phi_mm = np.load('mydata/djinn_true_1d/phi_mixed1.npy')
keff_mm = np.load('mydata/djinn_true_1d/keff_mixed1.npy')
_,_,_,_,_,scatter_mm,fission_mm,_,_,_ = r.problem.variables(0.05,ptype='mixed1',symm=True)

# phi_mm_scat = np.load('mydata/djinn_scatter_1d/phi_mixed1_reg_25.npy')
# keff_mm_scat = np.load('mydata/djinn_scatter_1d/keff_mixed1_reg_25.npy')
phi_mm_fis = np.load('mydata/djinn_fission_1d/phi_mixed1_reg_25.npy')
keff_mm_fis = np.load('mydata/djinn_fission_1d/keff_mixed1_reg_25.npy')
phi_mm_both = np.load('mydata/djinn_both_1d/phi_mixed1_reg_25.npy')
keff_mm_both = np.load('mydata/djinn_both_1d/keff_mixed1_reg_25.npy')

# %%
fig,ax1 = plt.subplots(figsize=(10,8))
ln1 = ax1.plot(space,sn.totalFissionRate(fission_mm[:,:low_ind,:low_ind], phi_mm[:,:low_ind]),alpha=0.6,c='k',label='True Thermal, keff {}'.format(np.round(keff_mm,5)))
ln2 = ax1.plot(space,sn.totalFissionRate(fission_mm[:,:low_ind,:low_ind], phi_mm_fis[:,:low_ind]),c='r',label='DJINN Fission Thermal, keff {}'.format(np.round(keff_mm_fis,5)),alpha=0.6)
ln3 = ax1.plot(space,sn.totalFissionRate(fission_mm[:,:low_ind,:low_ind], phi_mm_both[:,:low_ind]),c='b',label='DJINN Both Thermal, keff {}'.format(np.round(keff_mm_both,5)),alpha=0.6)

plt.grid(color='gray',linestyle='--'); ax1.set_xlabel('Location (cm)',fontsize=16)
ax1.set_ylabel('Thermal Fission Rate',fontsize=16); #plt.legend(loc='upper center',prop={'size':14})
plt.title(r'UH$_3$ 12% | 27% | 12% Enrichment',fontsize=18); ax1.tick_params(axis='both',labelsize=12)

ax2 = ax1.twinx()
ln4 = ax2.plot(space,sn.totalFissionRate(fission_mm[:,high_ind:,high_ind:], phi_mm[:,high_ind:]),'--',c='k',label='True Fast',alpha=0.6)
ln5 = ax2.plot(space,sn.totalFissionRate(fission_mm[:,high_ind:,high_ind:], phi_mm_fis[:,high_ind:]),c='r',ls='--',label='DJINN Fission Fast',alpha=0.6)
ln6 = ax2.plot(space,sn.totalFissionRate(fission_mm[:,high_ind:,high_ind:], phi_mm_both[:,high_ind:]),c='b',ls='--',label='DJINN Both Fast',alpha=0.6)


lns = ln1 + ln4 + ln2 + ln5 + ln3 + ln6
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left',prop={'size':14})

ax2.set_ylabel('Fast Fission Rate',fontsize=16,color='r')
ax2.tick_params(axis='y',labelsize=12,labelcolor='r')
# plt.savefig('paper_graphs/infinite27.png',bbox_inches='tight')
plt.show()
