# os.system('gcc -fPIC -shared -o discrete1/data/cfunctions.so discrete1/data/cfunctions.c')

import numpy as np
from .FSproblems import Selection

class Hybrid:
    def __init__(self,problem,G,N):
        """ G and N are lists of [uncollided,collided]  """
        
        self.problem = problem
        if type(G) is list:
            self.Gu = G[0]; self.Gc = G[1]
            # Have to call for splitting
            self.splits = Tools.energy_distribution(self.Gu,self.Gc)
            self.delta_u = Selection.energy_diff(self.problem,self.Gu)
            # Sum the delta_u
            self.delta_c = [sum(self.delta_u[ii]) for ii in self.splits]
        else:
            self.Gu = G; self.Gc = G

        if type(N) is list:
            self.Nu = N[0]; self.Nc = N[1]
        else:
            self.Nu = N; self.Nc = N

    def run(self):

        uncollided = Uncollided(*Selection.select(self.problem,self.Gu,self.Nu)[0])
        collided = Collided(*Selection.select(self.problem,self.Gc,self.Nc)[0])
        
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
            if self.Gu != self.Gc:
                source_c = Tools.big_2_small(source_c,self.delta_u,self.delta_c,self.splits)
            # Step 3: Solve Collided Equation
            phi_c = collided.multi_group(speed,source_c,phi_u)
            # Resize phi_c
            if self.Gu != self.Gc:
                phi = Tools.small_2_big(phi_c,self.delta_u,self.delta_c,self.splits) + phi_u
            else:
                phi = phi_c + phi_u
            # print('Combine collided and uncollided phi\t',np.sum(phi))
            # Step 4: Calculate next time step
            source = np.einsum('ijk,ik->ij',uncollided.fission,phi) + np.einsum('ijk,ik->ij',uncollided.scatter,phi) + uncollided.source
            phi,psi_next = uncollided.multi_group(psi_last,speed,source)

            # print(self.delta_u,self.delta_c)
            # print('Time {}\tPhi'.format(t),np.sum(phi))

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
            if np.isnan(change):
                change = 0
            # print('Uncollided Change is',change,'\n===================================')
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
            if np.isnan(change):
                change = 0
            # print('Collided Change is',change,'\n===================================')
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()

        return phi 


class Tools:

    def energy_distribution(big,small):
        """ List of slices for different energy sizes
        Arguments:
            big: uncollided energy groups, int
            small: collided energy groups, int
        Returns:
            list of slices   """
        import numpy as np

        new_grid = np.ones((small)) * int(big/small)
        new_grid[np.linspace(0,small-1,big % small,dtype=int)] += 1

        inds = np.cumsum(np.insert(new_grid,0,0),dtype=int)

        splits = [slice(ii,jj) for ii,jj in zip(inds[:small],inds[1:])]
        return splits

    def small_2_big(mult_c,delta_u,delta_c,splits):
        import numpy as np

        Gu = len(delta_u)
        size = (mult_c.shape[0],Gu)
        mult_u = np.zeros(size)
        factor = delta_u.copy()
        

        for count,index in enumerate(splits):
            for ii in np.arange(index.indices(Gu)[0],index.indices(Gu)[1]):
                mult_u[:,ii] = mult_c[:,count]
                factor[ii] /= delta_c[count]

        mult_u *= factor


        return mult_u

    def big_2_small(mult_u,delta_u,delta_c,splits):
        import numpy as np

        size = (mult_u.shape[0],len(delta_c))
        mult_c = np.zeros(size)

        for count,index in enumerate(splits):
            mult_c[:,count] = np.sum(mult_u[:,index],axis=1) 

        return mult_c



