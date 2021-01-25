# os.system('gcc -fPIC -shared -o discrete1/data/cfunctions.so discrete1/data/cfunctions.c')

from discrete1.setup_mg import selection

class Hybrid:
    def __init__(self,problem,G,N):
        """ G and N are lists of [uncollided,collided]  """
        self.problem = problem
        if list(G):
            self.Gu = G; self.Gc = G
        else:
            self.Gu = G[0]; self.Gc = G[1]

        if list(N):
            self.Nu = N; self.Nc = N
        else:
            self.Nu = N[0]; self.Nc = N[1]

    def run(self):
        from discrete1.setup_mg import selection
        import numpy as np

        uncollided = Uncollided(*selection(self.problem,self.Gu,self.Nu))
        collided = Collided(*selection(self.problem,self.Gc,self.Nc))

        print(uncollided)
        # T = 100; dt = 1; v = 1
        # psi_last = np.zeros((I,N,G))
        # speed = 1/(v*dt)
        # time_phi = []

        # for t in range(int(T/dt)):      
        #     # Step 1: Solve Uncollided Equation
        #     phi_u,_ = multigroup_uncollided(psi_last,speed,G,N,mu,w,total,source_u,I,inv_delta)
        #     # Step 2: Compute Source for Collided
        #     source_c = np.einsum('ijk,ik->ij',scatter,phi_u) + np.einsum('ijk,ik->ij',fission,phi_u)
        #     # Resizing
        #     # source_c = big_2_small(source_c,delta_u,delta_c,splits)
        #     # Step 3: Solve Collided Equation
        #     phi_c = multigroup_collided(speed,G,N,mu,w,total,scatter,source_c,I,inv_delta,phi_u)
        #     # Resize phi_c
        #     # phi = small_2_big(phi_c,delta_u,delta_c,splits) + phi_u
        #     phi = phi_c + phi_u
        #     # Step 4: Calculate next time step
        #     source = np.einsum('ijk,ik->ij',fission,phi) + np.einsum('ijk,ik->ij',scatter,phi) + source_u
        #     phi,psi_next = multigroup_uncollided(psi_last,speed,G,N,mu,w,total,source,I,inv_delta)
        #     psi_last = psi_next.copy()
        #     time_phi.append(phi)

        # return phi,time_phi

class Uncollided:
    def __init__(self,G,N,mu,w,total,scatter,fission,source,I,delta):
        self.G = G; self.N = N; 
        self.mu = mu; self.w = w
        self.total = total
        self.scatter = scatter
        self.fission = fission
        self.source = source
        self.I = I
        self.delta = delta

    def one_group(self,psi_last,speed,total_,source_):
        """ Step 1 of Hybrid
        Arguments:
            Different variables for collided and uncollided except I and inv_delta 
            psi_last: last time step, of size I x N
            speed: 1/(v*dt)   """
        import numpy as np
        import ctypes

        clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cfunctions.so')
        sweep = clibrary.uncollided

        phi = np.zeros((self.I),dtype='float64')

        psi_next = np.zeros(psi_last.shape,dtype='float64')

        weight = self.mu * self.delta

        for n in range(self.N):
            # Determine the direction
            direction = ctypes.c_int(int(np.sign(mu[n])))
            weight = np.sign(self.mu[n]) * self.mu[n] * self.delta
            # Collecting Angle
            psi_angle = np.zeros((self.I),dtype='float64')
            psi_ptr = ctypes.c_void_p(psi_angle.ctypes.data)
            # Source Terms
            rhs = (source_ + psi_last[:,n] * speed).astype('float64')
            rhs_ptr = ctypes.c_void_p(rhs.ctypes.data)

            top_mult = (weight - 0.5 * total_ - 0.5 * speed).astype('float64')
            top_ptr = ctypes.c_void_p(top_mult.ctypes.data)

            bottom_mult = (1/(0.5 * total_ + weight + 0.5 * speed)).astype('float64')
            bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)

            phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                
            sweep(phi_ptr,psi_ptr,rhs_ptr,top_ptr,bot_ptr,ctypes.c_double(w[n]),direction)

            psi_next[:,n] = psi_angle.copy()
        
        return phi,psi_next

    def multi_group(self,psi_last,speed):
        # G is Gu
        import numpy as np

        phi_old = np.random.rand(self.I,self.G)
        psi_next = np.zeros(psi_last.shape)

        tol = 1e-08; MAX_ITS = 100
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)
            for g in range(self.G):
                # print('Here')
                phi[:,g],psi_next[:,:,g] = Uncollided.one_group(psi_last[:,:,g],speed,self.total[:,g],self.source[:,g])
                # print('There')
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            # print('Change is',change,'\n===================================')
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 

            phi_old = phi.copy()
        return phi,psi_next


        

        
        

def one_group_uncollided(psi_last,speed,N,mu,w,total,source,I,inv_delta):
    """ Step 1 of Hybrid
    Arguments:
        Different variables for collided and uncollided except I and inv_delta 
        psi_last: last time step, of size I x N
        speed: 1/(v*dt)   """
    import numpy as np
    import ctypes

    clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cfunctions.so')
    sweep = clibrary.uncollided

    # no_scatter = np.zeros((I),dtype='float64')
    # ns_ptr = ctypes.c_void_p(no_scatter.ctypes.data)

    phi = np.zeros((I),dtype='float64')

    psi_next = np.zeros(psi_last.shape,dtype='float64')

    weight = mu * inv_delta

    for n in range(N):
        # Determine the direction
        direction = ctypes.c_int(int(np.sign(mu[n])))
        weight = np.sign(mu[n]) * mu[n] * inv_delta

        # Collecting Angle
        psi_angle = np.zeros((I),dtype='float64')
        psi_ptr = ctypes.c_void_p(psi_angle.ctypes.data)

        # Source Terms
        rhs = (source + psi_last[:,n] * speed).astype('float64')
        rhs_ptr = ctypes.c_void_p(rhs.ctypes.data)

        top_mult = (weight - 0.5 * total - 0.5 * speed).astype('float64')
        top_ptr = ctypes.c_void_p(top_mult.ctypes.data)

        bottom_mult = (1/(0.5 * total + weight + 0.5 * speed)).astype('float64')
        bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)

        phi_ptr = ctypes.c_void_p(phi.ctypes.data)
            
        sweep(phi_ptr,psi_ptr,rhs_ptr,top_ptr,bot_ptr,ctypes.c_double(w[n]),direction)

        psi_next[:,n] = psi_angle.copy()
    
    return phi,psi_next

def multigroup_uncollided(psi_last,speed,G,N,mu,w,total,source,I,inv_delta,tol=1e-08,MAX_ITS=100):
    # G is Gu
    import numpy as np

    phi_old = np.random.rand(I,G)

    psi_next = np.zeros(psi_last.shape)

    converged = 0; count = 1
    while not (converged):
        phi = np.zeros(phi_old.shape)
        for g in range(G):
            # print('Here')
            phi[:,g],psi_next[:,:,g] = one_group_uncollided(psi_last[:,:,g],speed,N,mu,w,total[:,g],source[:,g],I,inv_delta)
            # print('There')
        change = np.linalg.norm((phi - phi_old)/phi/(I))
        if np.isnan(change):
            change = 0
        # print('Change is',change,'\n===================================')
        count += 1
        converged = (change < tol) or (count >= MAX_ITS) 

        phi_old = phi.copy()
    return phi,psi_next

def update_q(scatter,phi,start,stop,g):
    import numpy as np
    return np.sum(scatter[:,g,start:stop]*phi[:,start:stop],axis=1)

def one_group_collided(speed,N,mu,w,total,scatter,source,I,inv_delta,guess,tol=1e-08,MAX_ITS=100):
    import numpy as np
    import ctypes

    clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cfunctions.so')
    sweep = clibrary.collided

    source = source.astype('float64')
    source_ptr = ctypes.c_void_p(source.ctypes.data)

    phi_old = guess.copy()
    # no_scatter = np.zeros((I),dtype='float64')
    # ns_ptr = ctypes.c_void_p(no_scatter.ctypes.data)

    converged = 0; count = 1
    while not (converged):
        phi = np.zeros((I),dtype='float64')
        for n in range(N):
            # Determine the direction
            direction = ctypes.c_int(int(np.sign(mu[n])))
            weight = np.sign(mu[n]) * mu[n] * inv_delta

            # Collecting Angle
            # psi_angle = np.zeros((I),dtype='float64')
            # psi_ptr = ctypes.c_void_p(psi_angle.ctypes.data)

            top_mult = (weight - 0.5 * total - 0.5 * speed).astype('float64')
            top_ptr = ctypes.c_void_p(top_mult.ctypes.data)

            bottom_mult = (1/(0.5 * total + weight + 0.5 * speed)).astype('float64') 
            bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)

            temp_scat = (scatter * phi_old).astype('float64')
            ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)

            phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                
            sweep(phi_ptr,ts_ptr,source_ptr,top_ptr,bot_ptr,ctypes.c_double(w[n]),direction)
        change = np.linalg.norm((phi - phi_old)/phi/(I))
        converged = (change < tol) or (count >= MAX_ITS) 
        count += 1
        phi_old = phi.copy()
    return phi

def multigroup_collided(speed,G,N,mu,w,total,scatter,source,I,inv_delta,guess,tol=1e-08,MAX_ITS=100):
    import numpy as np
    phi_old = guess.copy()
    # psi_last = np.zeros((N,I,G))

    converged = 0; count = 1
    while not (converged):
        phi = np.zeros(phi_old.shape)
        for g in range(G):
            q_tilde = source[:,g] + update_q(scatter,phi_old,g+1,G,g)
            if g != 0:
                q_tilde += update_q(scatter,phi,0,g,g)
            phi[:,g] = one_group_collided(speed,N,mu,w,total[:,g],scatter[:,g,g],q_tilde,I,inv_delta,phi_old[:,g])
        change = np.linalg.norm((phi - phi_old)/phi/(I))
        converged = (change < tol) or (count >= MAX_ITS) 
        count += 1
        phi_old = phi.copy()

    return phi #,psi_last



def driver():
    import numpy as np

    G,N,mu,w,total,scatter,fission,source_u,I,inv_delta = reeds(8,1000)
    delta_c = [1]; delta_u = [1]
    splits = [slice(0,1)]
    
    T = 100; dt = 1; v = 1
    psi_last = np.zeros((I,N,G))
    speed = 1/(v*dt)
    time_phi = []

    for t in range(int(T/dt)):      
        # Step 1: Solve Uncollided Equation
        phi_u,_ = multigroup_uncollided(psi_last,speed,G,N,mu,w,total,source_u,I,inv_delta)
        # Step 2: Compute Source for Collided
        source_c = np.einsum('ijk,ik->ij',scatter,phi_u) + np.einsum('ijk,ik->ij',fission,phi_u)
        # Resizing
        # source_c = big_2_small(source_c,delta_u,delta_c,splits)
        # Step 3: Solve Collided Equation
        phi_c = multigroup_collided(speed,G,N,mu,w,total,scatter,source_c,I,inv_delta,phi_u)
        # Resize phi_c
        # phi = small_2_big(phi_c,delta_u,delta_c,splits) + phi_u
        phi = phi_c + phi_u
        # Step 4: Calculate next time step
        source = np.einsum('ijk,ik->ij',fission,phi) + np.einsum('ijk,ik->ij',scatter,phi) + source_u
        phi,psi_next = multigroup_uncollided(psi_last,speed,G,N,mu,w,total,source,I,inv_delta)
        psi_last = psi_next.copy()
        time_phi.append(phi)

    return phi,time_phi
        
def small_2_big(mult_c,delta_u,delta_c,splits):
    import numpy as np

    mult_u = np.zeros((len(delta_u)))
    for count,index in enumerate(splits):
        mult_u[index] = mult_c[count]
        delta_u[index] /= delta_c[count]

    mult_u *= delta_u

    return mult_u

def big_2_small(mult_u,delta_u,delta_c,splits):
    import numpy as np

    mult_c = np.zeros((len(delta_c)))
    for count,index in enumerate(splits):
        mult_c[count] = np.sum(mult_u[splits]) 

    return mult_c

def energy_splits_delta(Gc,energy_u=None,delta=False):
    import numpy as np
    if energy_u is None:
        energy_ = np.load('discrete1/data/energyGrid.npy')
    Gu = len(energy_u) - 1
    split = int(Gu / Gc); rmdr = Gu % Gc

    new_grid = np.ones(Gc) * split
    new_grid[np.linspace(0,Gc-1,rmdr,dtype=int)] += 1

    inds = np.cumsum(np.insert(new_grid,0,0),dtype=int)

    energy_c = energy_u[inds]
    splits = [slice(ii,jj) for ii,jj in zip(inds[:len(inds)-1],inds[1:])]

    if delta:
        delta_u = np.diff(energy_u); delta_c = np.diff(energy_c)
        return delta_u,delta_c,splits

    return energy_c,splits

def resizer(energy_u):
    # energy_u = np.load('discrete1/data/energyGrid.npy')
    
    # Gu = len(energy_u) - 1
    # Gc = 40 # Given
    
    # split = int(Gu / Gc); rmdr = Gu % Gc
    # new_grid = np.ones(Gc) * split
    # new_grid[np.linspace(0,Gc-1,rmdr,dtype=int)] += 1
    
    # inds = np.cumsum(np.insert(new_grid,0,0),dtype=int)
    
    # energy_c = energy_u[inds]
    
    # delta_Eu = np.diff(energy_u)
    # delta_Ec = np.diff(energy_c)
    
    # splits = [slice(ii,jj) for ii,jj in zip(inds[:len(inds)-1],inds[1:])]
    
    # Small to Big
    # smult_c = np.random.rand(Gc) # Given
    # smult_u = np.zeros((Gu))
    
    # for count,index in enumerate(splits):
    #     smult_u[index] = smult_c[count]
    #     delta_Eu[index] /= delta_Ec[count]
    
    # smult_u *= delta_Eu
    
    # # Big to Small
    # fmult_u = np.random.rand(Gu)
    # fmult_c = np.zeros((Gc))
    
    # for count,index in enumerate(splits):
    #     fmult_c[count] = np.sum(fmult_u[splits]) 
    return None


# G,N,mu,w,total,scatter,fission,source_u,I,inv_delta = reeds(8,1000)
# print(total.shape)
# print(source_u.shape)

import matplotlib.pyplot as plt
import numpy as np

prob = Hybrid('reeds',[2,2],[8,8])
prob.run()

# phi,time_phi = driver()

# for ii in range(len(time_phi)):
#     print('TS {}\tSum: {}'.format(ii,np.sum(time_phi[ii])))

# plt.plot(np.linspace(0,16,1000),phi)
# plt.grid()
# plt.show()




# def reeds(N,I):
#     import numpy as np

#     G = 1; L = 0; R = 16.
#     mu,w = np.polynomial.legendre.leggauss(N)
#     w /= np.sum(w); #N = int(0.5*N)
#     # w = w[N:]; mu = mu[N:] # Symmetry
    
#     delta = R/I

#     # boundaries = [slice(0,int(2/delta)),slice(int(2/delta),int(4/delta)),
#     #     slice(int(4/delta),int(5/delta)),slice(int(5/delta),int(6/delta)),
#     #     slice(int(6/delta),int(8/delta))]

#     boundaries = [slice(0,int(2/delta)),slice(int(2/delta),int(3/delta)),
#         slice(int(3/delta),int(5/delta)),slice(int(5/delta),int(6/delta)),
#         slice(int(6/delta),int(10/delta)),slice(int(10/delta),int(11/delta)),
#         slice(int(11/delta),int(13/delta)),slice(int(13/delta),int(14/delta)),
#         slice(int(14/delta),int(16/delta))]
    
#     # print(boundaries) 

#     total_ = np.zeros((I)); total_vals = [10,10,0,5,50,5,0,10,10]
#     scatter_ = np.zeros((I)); scatter_vals = [9.9,9.9,0,0,0,0,0,9.9,9.9]
#     source_ = np.zeros((I)); source_vals = [0,1,0,0,50,0,0,1,0]

#     # print(len(boundaries),len(total_vals),len(scatter_vals),len(source_vals))

#     for ii in range(len(boundaries)):
#         total_[boundaries[ii]] = total_vals[ii]
#         scatter_[boundaries[ii]] = scatter_vals[ii]
#         source_[boundaries[ii]] = source_vals[ii]

#     fission_ = np.zeros((scatter_.shape))

#     return G,N,mu,w,total_[:,None],scatter_[:,None,None],fission_[:,None,None],source_[:,None],I,1/delta

# def multi_example(N,I):
#     import numpy as np

#     G = 4; L = 0; R = 5.
#     mu,w = np.polynomial.legendre.leggauss(N)
#     w /= np.sum(w); N = int(0.5*N)
#     w = w[N:]; mu = mu[N:] # Symmetry


#     delta = R/I
#     sigma_a = np.array([0.00490, 0.00280, 0.03050, 0.12100])
#     sigma_ds = np.array([0.08310,0.05850,0.06510])
#     D_g = np.array([2.16200,1.08700,0.63200,0.35400])
        
#     total_ = np.tile(1/(3*D_g),(I,1))
#     down_scat = np.array([(1/(3*D_g[ii]) - sigma_a[ii]) - sigma_ds[ii] for ii in range(G-1)])

#     scatter_vals = np.diag(down_scat,-1)
#     np.fill_diagonal(scatter_vals,1/(3*D_g) - sigma_a)
#     scatter_ = np.tile(scatter_vals,(I,1,1))

#     source_vals = [1e12,0,0,0]
#     source_ = np.tile(source_vals,(I,1))

#     fission_ = np.zeros((scatter_.shape))

#     return G,N,mu,w,total_,scatter_,fission_,source_,I,1/delta