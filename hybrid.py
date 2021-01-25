# os.system('gcc -fPIC -shared -o discrete1/data/cfunctions.so discrete1/data/cfunctions.c')

def reeds(N,I):
    import numpy as np

    G = 1; L = 0; R = 16.
    mu,w = np.polynomial.legendre.leggauss(N)
    w /= np.sum(w); #N = int(0.5*N)
    # w = w[N:]; mu = mu[N:] # Symmetry
    
    delta = R/I

    # boundaries = [slice(0,int(2/delta)),slice(int(2/delta),int(4/delta)),
    #     slice(int(4/delta),int(5/delta)),slice(int(5/delta),int(6/delta)),
    #     slice(int(6/delta),int(8/delta))]

    boundaries = [slice(0,int(2/delta)),slice(int(2/delta),int(4/delta)),
        slice(int(4/delta),int(5/delta)),slice(int(5/delta),int(6/delta)),
        slice(int(6/delta),int(10/delta)),slice(int(10/delta),int(11/delta)),
        slice(int(11/delta),int(12/delta)),slice(int(12/delta),int(14/delta)),
        slice(int(14/delta),int(16/delta))]
    
    print(boundaries) 

    total_ = np.zeros((I)); total_vals = [1,1,0,5,50,5,0,1,1]
    scatter_ = np.zeros((I)); scatter_vals = [0.9,0.9,0,0,0,0,0,0.9,0.9]
    source_ = np.zeros((I)); source_vals = [0,1,0,0,50,0,0,1,0]

    print(len(boundaries),len(total_vals),len(scatter_vals),len(source_vals))

    for ii in range(len(boundaries)):
        total_[boundaries[ii]] = total_vals[ii]
        scatter_[boundaries[ii]] = scatter_vals[ii]
        source_[boundaries[ii]] = source_vals[ii]

    fission_ = np.zeros((scatter_.shape))

    return G,N,mu,w,total_[:,None],scatter_[:,None,None],fission_[:,None,None],source_,I,1/delta

def multi_example(N,I):
    import numpy as np

    G = 4; L = 0; R = 5.
    mu,w = np.polynomial.legendre.leggauss(N)
    w /= np.sum(w); N = int(0.5*N)
    w = w[N:]; mu = mu[N:] # Symmetry


    delta = R/I
    sigma_a = np.array([0.00490, 0.00280, 0.03050, 0.12100])
    sigma_ds = np.array([0.08310,0.05850,0.06510])
    D_g = np.array([2.16200,1.08700,0.63200,0.35400])
        
    total_ = np.tile(1/(3*D_g),(I,1))
    down_scat = np.array([(1/(3*D_g[ii]) - sigma_a[ii]) - sigma_ds[ii] for ii in range(G-1)])

    scatter_vals = np.diag(down_scat,-1)
    np.fill_diagonal(scatter_vals,1/(3*D_g) - sigma_a)
    scatter_ = np.tile(scatter_vals,(I,1,1))

    source_vals = [1e12,0,0,0]
    source_ = np.tile(source_vals,(I,1))

    fission_ = np.zeros((scatter_.shape))

    return G,N,mu,w,total_,scatter_,fission_,source_,I,1/delta

def one_group_collided(N,mu,w,total,scatter,source,I,inv_delta,guess,tol=1e-08,MAX_ITS=100):
    import numpy as np
    import ctypes

    clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cfunctions.so')
    sweep = clibrary.sweep

    speed = 0
    source = source.astype('float64')
    source_ptr = ctypes.c_void_p(source.ctypes.data)

    phi_old = guess.copy()
    # no_scatter = np.zeros((I),dtype='float64')
    # ns_ptr = ctypes.c_void_p(no_scatter.ctypes.data)

    converged = 0; count = 1
    while not (converged):
        phi = np.zeros((I),dtype='float64')
        for n in range(N):
            weight = mu[n]*inv_delta

            top_mult = (weight - 0.5*total - 0.5*speed).astype('float64')
            top_ptr = ctypes.c_void_p(top_mult.ctypes.data)

            bottom_mult = (1/(0.5*total + 0.5*speed + weight)).astype('float64')
            bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)

            temp_scat = (scatter * phi_old).astype('float64')
            ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)

            phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                
            sweep(phi_ptr,ts_ptr,source_ptr,top_ptr,bot_ptr,ctypes.c_double(w[n]))
        change = np.linalg.norm((phi - phi_old)/phi/(I))
        converged = (change < tol) or (count >= MAX_ITS) 
        count += 1
        phi_old = phi.copy()
    return phi

def one_group_uncollided(N,mu,w,total,source,I,inv_delta):
    import numpy as np
    import ctypes

    clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cfunctions.so')
    sweep = clibrary.uncollided

    speed = 0
    source = source.astype('float64')
    source_ptr = ctypes.c_void_p(source.ctypes.data)

    # no_scatter = np.zeros((I),dtype='float64')
    # ns_ptr = ctypes.c_void_p(no_scatter.ctypes.data)

    phi = np.zeros((I),dtype='float64')

    for n in range(N):


        # Back and forth sweeping angular flux
        # psi_front = np.zeros((I),dtype='float64')
        # frt_ptr = ctypes.c_void_p(psi_front.ctypes.data)
        # psi_back = np.zeros((I),dtype='float64')
        # bck_ptr = ctypes.c_void_p(psi_back.ctypes.data)

        weight = mu[n]*inv_delta

        top_mult = (weight - 0.5*total).astype('float64')
        top_ptr = ctypes.c_void_p(top_mult.ctypes.data)

        bottom_mult = (1/(0.5*total + weight)).astype('float64')
        bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)

        phi_ptr = ctypes.c_void_p(phi.ctypes.data)
            
        sweep(phi_ptr,source_ptr,top_ptr,bot_ptr,frt_ptr,bck_ptr,ctypes.c_double(w[n]))

        psi_last[n] = psi_front.copy()
        psi_last[2*N-1-n] = psi_back.copy()
    
    return phi

def one_group_collided_s(N,mu,w,total,scatter,source,I,inv_delta,guess,tol=1e-08,MAX_ITS=100):
    import numpy as np

    phi_old = guess.copy()
    psi_last = np.zeros((N,I))

    converged = 0; count = 1
    while not (converged):
        phi = np.zeros((I),dtype='float64')
        for n in range(N):
            if mu[n] > 0:
                psi_bottom = 0
                for ii in range(I):
                    psi_top = (scatter[ii]*phi_old[ii] + source[ii] + psi_bottom*(mu[n]*inv_delta - 0.5*total[ii]))/(0.5*total[ii] + mu[n]*inv_delta)
                    # psi_last[n,ii] = 0.5*(psi_top+psi_bottom)
                    phi[ii] = phi[ii] + w[n]*0.5*(psi_top+psi_bottom)
                    psi_last[n,ii] = 0.5*(psi_top+psi_bottom)
                    psi_bottom = psi_top
            elif mu[n] < 0:
                psi_top = 0
                for ii in range(I-1,-1,-1):
                    psi_bottom = (scatter[ii]*phi_old[ii] + source[ii] + psi_top*(-mu[n]*inv_delta - 0.5*total[ii]))/(0.5*total[ii] - mu[n]*inv_delta)
                    phi[ii] = phi[ii] + w[n]*0.5*(psi_top+psi_bottom)
                    psi_last[n,ii] = 0.5*(psi_top+psi_bottom)
                    psi_top = psi_bottom
        change = np.linalg.norm((phi - phi_old)/phi/(I))
        converged = (change < tol) or (count >= MAX_ITS) 
        count += 1
        phi_old = phi.copy()
    return phi,psi_last

def one_group_uncollided_s(N,mu,w,total,source,I,inv_delta,psi_last,dt):
    import numpy as np
 
    speed = 1/(dt)
    phi = np.zeros((I),dtype='float64')

    for n in range(N):
        if mu[n] > 0:
            psi_bottom = 0
            for ii in range(I):
                psi_top = (source[ii] + psi_last[n,ii]*speed + psi_bottom*(mu[n]*inv_delta - 0.5*total[ii]))/(0.5*total[ii] + mu[n]*inv_delta)
                # psi_last[n,ii] = 0.5*(psi_top+psi_bottom)
                phi[ii] = w[n]*0.5*(psi_top+psi_bottom)
                psi_bottom = psi_top
        elif mu[n] < 0:
            psi_top = 0
            for ii in range(I-1,-1,-1):
                psi_bottom = (source[ii] + psi_last[n,ii]*speed + psi_top*(-mu[n]*inv_delta - 0.5*total[ii]))/(0.5*total[ii] - mu[n]*inv_delta)
                phi[ii] = w[n]*0.5*(psi_top+psi_bottom)
                psi_top = psi_bottom
    print(np.sum(phi))
    return phi

def multigroup_uncollided_s(G,N,mu,w,total,source,I,inv_delta,psi_last,dt,tol=1e-08,MAX_ITS=100):
    # G is Gu
    import numpy as np

    phi_old = np.random.rand(I,G)

    converged = 0; count = 1
    while not (converged):
        phi = np.zeros(phi_old.shape)
        for g in range(G):
            phi[:,g] = one_group_uncollided_s(N,mu,w,total,source,I,inv_delta,psi_last[:,:,g],dt)
        change = np.linalg.norm((phi - phi_old)/phi/(I))
        if np.isnan(change):
            change = 0
        # print('Change is',change,'\n===================================')
        count += 1
        converged = (change < tol) or (count >= MAX_ITS) 

        phi_old = phi.copy()

    return phi

def update_q(scatter,phi,start,stop,g):
    import numpy as np
    return np.sum(scatter[:,g,start:stop]*phi[:,start:stop],axis=1)

def multigroup_collided_s(G,N,mu,w,total,scatter,source,I,inv_delta,guess,tol=1e-08,MAX_ITS=100):
    import numpy as np
    phi_old = guess.copy()
    psi_last = np.zeros((N,I,G))
    converged = 0; count = 1
    while not (converged):
        phi = np.zeros(phi_old.shape)
        for g in range(G):
            q_tilde = source[:,g] + update_q(scatter,phi_old,g+1,G,g)
            if g != 0:
                q_tilde += update_q(scatter,phi,0,g,g)
            phi[:,g],psi_last[:,:,g] = one_group_collided_s(N,mu,w,total[:,g],scatter[:,g,g],q_tilde,I,inv_delta,phi_old[:,g])
        change = np.linalg.norm((phi - phi_old)/phi/(I))
        converged = (change < tol) or (count >= MAX_ITS) 
        count += 1
        phi_old = phi.copy()

    return phi,psi_last

def time_step_update(G,N,mu,w,total,scatter,source,I,inv_delta,guess,tol=1e-08,MAX_ITS=2):
    import numpy as np
    
    psi_last = np.zeros((N,I,G))
    phi_old = guess.copy()

    converged = 0; count = 1
    while not (converged):
        phi = np.zeros(phi_old.shape)
        for g in range(G):
            # q_tilde = source[:,g] + update_q(scatter,phi_old,g+1,G,g)
            # if g != 0:
            #     q_tilde += update_q(scatter,phi,0,g,g)
            phi[:,g],psi_last[:,:,g] = one_group_collided_s(N,mu,w,total[:,g],scatter[:,g,g],source[:,g],I,inv_delta,np.zeros((I)))
        change = np.linalg.norm((phi - phi_old)/phi/(I))
        print(change)
        converged = (change < tol) or (count >= MAX_ITS) 
        count += 1
        phi_old = phi.copy()

    return phi,psi_last

def driver():
    import numpy as np

    G,N,mu,w,total,scatter,fission,source_u,I,inv_delta = reeds(8,400)
    delta_c = [1]; delta_u = [1]
    splits = [slice(0,1)]
    
    T = 0.05; dt = 0.01
    time_phi = []
    psi_last = np.zeros((N,I,G))
    for t in range(int(T/dt)):      
        # Step 1: Solve Uncollided Equation
        # phi_u is of size (I x Gu)
        phi_u = multigroup_uncollided_s(G,N,mu,w,total,source_u,I,inv_delta,psi_last,dt)
        # Step 2: Compute Source for Collided
        source_c = np.einsum('ijk,ik->ij',scatter,phi_u) + np.einsum('ijk,ik->ij',fission,phi_u)
        # Resizing
        # source_c = big_2_small(source_c,delta_u,delta_c,splits)
        # Step 3: Solve Collided Equation
        phi_c,_ = multigroup_collided_s(G,N,mu,w,total,scatter,source_c,I,inv_delta,phi_u)
        # Resize phi_c
        # phi = small_2_big(phi_c,delta_u,delta_c,splits) + phi_u
        phi = phi_c + phi_u
        # Step 4: Calculate next time step
        source = np.einsum('ijk,ik->ij',fission,phi) + source_u
        phi_,psi_last = time_step_update(G,N,mu,w,total,scatter,source,I,inv_delta,phi) 
        time_phi.append(phi_)
    return phi_,time_phi
        



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


# import numpy as np
# import matplotlib.pyplot as plt
# import ctypes

# G,N,mu,w,total,scatter,fission,source_u,I,inv_delta = reeds(8,1000)
# # delta_c = [1]; delta_u = [1]
# # splits = [slice(0,1)]

# clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/testingc.so')
# sweep = clibrary.uncollided

# speed = 0
# source_u = source_u.astype('float64')
# source_ptr = ctypes.c_void_p(source_u.ctypes.data)
# mem = ctypes.POINTER(ctypes.c_ubyte)()

# # test = ctypes.POINTER(source_u.ctypes.data)

# psi_ = np.zeros((N,I),dtype='float64')
# # psi_ptr = ctypes.c_void_p(ctypes.c_void_p(psi_.ctypes.data).ctypes.data)

# phi = np.zeros((I),dtype='float64')
# phi_ptr = ctypes.c_void_p(phi.ctypes.data)

# total = total.astype('float64')
# xs_ptr = ctypes.c_void_p(total.ctypes.data)

# w = w.astype('float64')
# w_ptr = ctypes.c_void_p(w.ctypes.data)

# mu = mu.astype('float64')
# mu_ptr = ctypes.c_void_p(mu.ctypes.data)

# N_ptr = ctypes.c_double(N)
# de_ptr = ctypes.c_double(inv_delta)

# sweep(phi_ptr,xs_ptr,ctypes.byref(mem),source_ptr,w_ptr,mu_ptr,N_ptr,de_ptr)

# # uncollided(void *flux, void *xs, void **psi, void *external, void *w, void *mu, double N, double delta)

# print('Populate Flux',np.sum(phi))
# print('Populate Angle',np.sum(psi_))

# phi_u = multigroup_uncollided(G,N,mu,w,total,source_u,I,inv_delta)
# # Step 2: Compute Source for Collided
# source_c = np.einsum('ijk,ik->ij',scatter,phi_u) + np.einsum('ijk,ik->ij',fission,phi_u)
# # Resizing
# # source_c = big_2_small(source_c,delta_u,delta_c,splits)
# # Step 3: Solve Collided Equation
# phi_c = multigroup_collided(G,N,mu,w,total,scatter,source_c,I,inv_delta,phi_u)
# # Resize phi_c
# # phi = small_2_big(phi_c,delta_u,delta_c,splits) + phi_u

# phi = phi_u + phi_c
# plt.plot(phi)

import matplotlib.pyplot as plt


# G,N,mu,w,total,scatter,fission,source_u,I,inv_delta = reeds(8,32)
# print(total)

phi,alls = driver()
for ii in range(len(alls)):
    plt.plot(alls[ii])
    plt.title('Number {}'.format(ii))
    # plt.savefig('../Desktop/image_{}.png'.format(str(ii).zfill(3)),bbox_inches='tight')
    plt.show()
