# os.system('gcc -fPIC -shared -o discrete1/data/cfunctions.so discrete1/data/cfunctions.c')

class Hybrid:
    def __init__(self,problem,G,N):
        """ G and N are lists of [uncollided,collided]  """
        self.problem = problem
        if type(G) is list:
            self.Gu = G[0]; self.Gc = G[1]
            # Have to call for splitting
        else:
            self.Gu = G; self.Gc = G

        if type(N) is list:
            self.Nu = N[0]; self.Nc = N[1]
        else:
            self.Nu = N; self.Nc = N

    def run(self):
        from .setup_mg import selection
        import numpy as np

        uncollided = Uncollided(*selection(self.problem,self.Gu,self.Nu))
        collided = Collided(*selection(self.problem,self.Gc,self.Nc))

        delta_u = [1]; delta_c = [1]
        splits = [slice(0,1)]
        
        T = 100; dt = 1; v = 1
        psi_last = np.zeros((uncollided.I,uncollided.N,uncollided.G))
        speed = 1/(v*dt)
        time_phi = []

        for t in range(int(T/dt)):      
            # Step 1: Solve Uncollided Equation
            phi_u,_ = uncollided.multi_group(psi_last,speed)
            # Step 2: Compute Source for Collided
            source_c = np.einsum('ijk,ik->ij',uncollided.scatter,phi_u) + np.einsum('ijk,ik->ij',uncollided.fission,phi_u)
            # Resizing
            # source_c = big_2_small(source_c,delta_u,delta_c,splits)
            # Step 3: Solve Collided Equation
            phi_c = collided.multi_group(speed,source_c,phi_u)
            # Resize phi_c
            # phi = small_2_big(phi_c,delta_u,delta_c,splits) + phi_u
            phi = phi_c + phi_u
            # Step 4: Calculate next time step
            source = np.einsum('ijk,ik->ij',uncollided.fission,phi) + np.einsum('ijk,ik->ij',uncollided.scatter,phi) + uncollided.source
            phi,psi_next = uncollided.multi_group(psi_last,speed,source)
            
            psi_last = psi_next.copy()
            time_phi.append(phi)

        return phi,time_phi

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

        clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cHybrid.so')
        sweep = clibrary.uncollided

        phi = np.zeros((self.I),dtype='float64')

        psi_next = np.zeros(psi_last.shape,dtype='float64')

        weight = self.mu * self.delta

        for n in range(self.N):
            # Determine the direction
            direction = ctypes.c_int(int(np.sign(self.mu[n])))
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
                
            sweep(phi_ptr,psi_ptr,rhs_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]),direction)

            psi_next[:,n] = psi_angle.copy()
        
        return phi,psi_next

    def multi_group(self,psi_last,speed,source=None):
        # G is Gu
        import numpy as np

        phi_old = np.random.rand(self.I,self.G)
        psi_next = np.zeros(psi_last.shape)

        if source is None:
            current = self.source.copy()
        else:
            current = source.copy()

        tol = 1e-08; MAX_ITS = 100
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)
            for g in range(self.G):
                phi[:,g],psi_next[:,:,g] = Uncollided.one_group(self,psi_last[:,:,g],speed,self.total[:,g],current[:,g])
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            # print('Change is',change,'\n===================================')
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 

            phi_old = phi.copy()
        return phi,psi_next


class Collided:
    def __init__(self,G,N,mu,w,total,scatter,fission,source,I,delta):
        self.G = G; self.N = N; 
        self.mu = mu; self.w = w
        self.total = total
        self.scatter = scatter
        self.fission = fission
        # self.source = source
        self.I = I
        self.delta = delta

    def one_group(self,speed,total_,scatter_,source_,guess_):
        import numpy as np
        import ctypes

        clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cHybrid.so')
        sweep = clibrary.collided

        source_ = source_.astype('float64')
        source_ptr = ctypes.c_void_p(source_.ctypes.data)

        phi_old = guess_.copy()

        tol = 1e-08; MAX_ITS = 100
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros((self.I),dtype='float64')
            for n in range(self.N):
                # Determine the direction
                direction = ctypes.c_int(int(np.sign(self.mu[n])))
                weight = np.sign(self.mu[n]) * self.mu[n] * self.delta

                top_mult = (weight - 0.5 * total_ - 0.5 * speed).astype('float64')
                top_ptr = ctypes.c_void_p(top_mult.ctypes.data)

                bottom_mult = (1/(0.5 * total_ + weight + 0.5 * speed)).astype('float64') 
                bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)

                temp_scat = (scatter_ * phi_old).astype('float64')
                ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)

                phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                    
                sweep(phi_ptr,ts_ptr,source_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]),direction)

            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi

    def update_q(self,phi,start,stop,g):
        import numpy as np
        return np.sum(self.scatter[:,g,start:stop]*phi[:,start:stop],axis=1)

    def multi_group(self,speed,source,guess):
        import numpy as np

        phi_old = guess.copy()
        
        tol = 1e-08; MAX_ITS = 100
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)
            for g in range(self.G):
                q_tilde = source[:,g] + Collided.update_q(self,phi_old,g+1,self.G,g)
                if g != 0:
                    q_tilde += Collided.update_q(self,phi,0,g,g)
                phi[:,g] = Collided.one_group(self,speed,self.total[:,g],self.scatter[:,g,g],q_tilde,phi_old[:,g])
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()

        return phi 





class Tools:

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
