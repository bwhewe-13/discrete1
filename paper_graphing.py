import numpy as np
import matplotlib.pyplot as plt
from discrete1.util import display

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



