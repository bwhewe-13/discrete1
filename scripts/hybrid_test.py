
import matplotlib.pyplot as plt
import numpy as np
from discrete1.hybrid import Hybrid
import time

# Original S8
start_8 = time.time()
prob = Hybrid('reeds',1,8)
phi_8,_ = prob.run()
end_8 = time.time()

# Reduced S2
start_2 = time.time()
prob = Hybrid('reeds',1,[8,2])
phi_2,_ = prob.run()
end_2 = time.time()

plt.plot(np.linspace(0,16,1000),phi_8,label='S8, Time: {}'.format(np.round(end_8 - start_8,3)),c='k',ls='--'); 
plt.plot(np.linspace(0,16,1000),phi_2,label='S2, Time: {}'.format(np.round(end_2 - start_2,3)),c='r',alpha=0.6); 
plt.legend(loc='best'); plt.grid()
plt.title('Comparing S8 and S2 Models')
plt.savefig('../../Desktop/reed.png',bbox_inches='tight')

# from discrete1.correct import Source
# from discrete1.setup import problem1

# problem = Source(*problem1.variables(0.15,'carbon'),)