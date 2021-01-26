
import matplotlib.pyplot as plt
import numpy as np
from discrete1.hybrid import Hybrid
from discrete1.source import Source
# from discrete1.util import sn
from discrete1.setup_mg import Reeds,FourGroup
import time, json

prob = Hybrid('reeds',1,8)
phi_true,_ = prob.run()

prob = Hybrid('reeds',4,8)
phi_approx,_ = prob.run()

# prob = Source(*FourGroup(4,8).variables())
# prob = Source(*Reeds(4,8).variables())
# prob.problem_information()
# phi_approx = prob.multi_group()

# for ii in range(4):
    # plt.plot(np.linspace(0,16,1000),phi_[:,ii],label='Group {}'.format(ii),alpha=0.6); 

plt.plot(np.linspace(0,16,1000),phi_true,label='True',c='k',ls='--'); 
plt.plot(np.linspace(0,16,1000),np.sum(phi_approx,axis=1),label='Approx',c='r',alpha=0.6); 
# plt.plot(np.linspace(0,5,1000),phi_,alpha=0.6); 
plt.legend(loc='best'); plt.grid()
# plt.title('Four Group Model')
plt.title('Reed')
# plt.savefig('../../Desktop/reed2.png',bbox_inches='tight')


# # Original S8
# for uncoll in [2,4,6,8]:
#     for coll in [2,4,6,8]:
#         mydict = {}
#         prob = Hybrid('reeds',1,[uncoll,coll])
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

#         with open('mydata/hybrid/reed_s{}_s{}.json'.format(uncoll,coll),'w') as fp:
#             json.dump(mydict,fp)
#         del phi,mydict,prob,times




