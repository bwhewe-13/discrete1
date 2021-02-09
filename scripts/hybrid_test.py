
import matplotlib.pyplot as plt
import numpy as np
from discrete1.hybrid import Hybrid
from discrete1.source import Source
# from discrete1.util import sn
from discrete1.setup_mg import Reeds #,FourGroup
import time, json

# print('Original')
# prob = Hybrid('reeds',1,8)
# phi_true,_ = prob.run()

# print('\nMultiple Collided')
# prob = Hybrid('reeds',[5,2],[8,2])
# phi_approx,_ = prob.run()

# plt.plot(np.linspace(0,16,1000),phi_true,label='True',c='k',ls='--'); 
# plt.plot(np.linspace(0,16,1000),np.sum(phi_approx,axis=1),label='Approx',c='r',alpha=0.6); 
# plt.legend(loc='best'); plt.grid()
# plt.title('Reed')
# plt.savefig('../Desktop/reed2.png',bbox_inches='tight')


# # Original S8
# for uncoll in [2,4,6,8]:
#     for coll in [2,4,6,8]:
#         mydict = {}
#         prob = Hybrid('reeds',[5,2],[uncoll,coll])
#         phi,_ = prob.run()
#         mydict['phi'] = phi.tolist()
#         times = []
#         for ii in range(10):
#             start = time.time()
#             phi,_ = prob.run()
#             end = time.time()
#             times.append(end-start)
#         times = np.mean(np.array(times))
#         mydict['time'] = times

#         with open('mydata/hybrid/reed_g5_g2_s{}_s{}.json'.format(uncoll,coll),'w') as fp:
#             json.dump(mydict,fp)
#         del phi,mydict,prob,times



import numpy as np
import json, glob
import matplotlib.pyplot as plt

g5g2 = np.sort(glob.glob('mydata/hybrid/reed_g5*'))

# print(g5g2)

time_g5g2 = {}
phi_g5g2 = {}

for add in g5g2:
    temp = json.load(open(add,'r'))
    label = add.split('g2_')[1].split('.json')[0]
    time_g5g2[label] = temp["time"]
    phi_g5g2[label] = temp["phi"]
    del temp,label
    