"""
Running Hybrid Problems (time dependent)
"""

from .fixed import FixedSource

import numpy as np
import ctypes

class Hybrid:
    def __init__(self,ptype,G,N):
        """ G and N are lists of [uncollided,collided]  """
        
        self.ptype = ptype
        self.geometry = 'slab'

        if type(G) is list:
            self.Gu = G[0]; self.Gc = G[1]
        else:
            self.Gu = G; self.Gc = G

        if type(N) is list:
            self.Nu = N[0]; self.Nc = N[1]
        else:
            self.Nu = N; self.Nc = N

    @classmethod
    def run(cls,ptype,G,N,T,dt,**kwargs):
        prob = cls(ptype,G,N)
        prob.T = T; prob.dt = dt
        if 'edges' in kwargs:
            prob.edges = kwargs['edges']
        else:
            prob.edges = None
        if 'enrich' in kwargs:
            prob.enrich = kwargs['enrich']
        else:
            prob.enrich = None
        if 'geometry' in kwargs:
            prob.geometry = kwargs['geometry']
        else:
            prob.geometry = 'slab'
        if 'td' in kwargs:
            prob.td = kwargs['td']
            if prob.td == 'BE':
                return prob.backward_euler()
            elif prob.td == 'BDF2':
                return prob.bdf2()


        return prob.backward_euler()

    def backward_euler(self):

        if self.enrich:
            un_attr,un_keys = FixedSource.initialize(self.ptype,self.Gu,self.Nu,T=self.T,dt=self.dt,hybrid=self.Gu,enrich=self.enrich,edges=self.edges)
            uncollided = Uncollided(*un_attr)
            col_attr,col_keys = FixedSource.initialize(self.ptype,self.Gc,self.Nc,T=self.T,dt=self.dt,hybrid=self.Gu,enrich=self.enrich,edges=self.edges)
            collided = Collided(*col_attr)
        else:
            un_attr,un_keys = FixedSource.initialize(self.ptype,self.Gu,self.Nu,T=self.T,dt=self.dt,hybrid=self.Gu,edges=self.edges)
            uncollided = Uncollided(*un_attr)
            col_attr,col_keys = FixedSource.initialize(self.ptype,self.Gc,self.Nc,T=self.T,dt=self.dt,hybrid=self.Gu,edges=self.edges)
            collided = Collided(*col_attr)
            
        time_phi = []
        speed_u = 1/(un_keys['v']*un_keys['dt']); speed_c = 1/(col_keys['v']*col_keys['dt'])
        phi_c = np.zeros((collided.I,collided.G))

        print('Initial Source ',np.sum(uncollided.lhs))
        steps = int(un_keys['T']/un_keys['dt'])

        # Initialize psi to zero
        psi_last = np.zeros((uncollided.I,uncollided.N,uncollided.G))
        for t in range(int(un_keys['T']/un_keys['dt'])): 
            # Step 1: Solve Uncollided Equation
            phi_u,_ = uncollided.multi_group(psi_last,speed_u,self.geometry)
            # Step 2: Compute Source for Collided
            source_c = np.einsum('ijk,ik->ij',uncollided.scatter,phi_u) + np.einsum('ijk,ik->ij',uncollided.fission,phi_u) 
            # Resizing
            if self.Gu != self.Gc:
                source_c = Tools.big_2_small(source_c,un_keys['delta_e'],col_keys['delta_e'],col_keys['splits'])
            # Step 3: Solve Collided Equation
            phi_c = collided.multi_group(speed_c,source_c,phi_c,self.geometry)
            # Resize phi_c
            if self.Gu != self.Gc:
                # phi = Tools.small_2_big(phi_c,self.delta_u,self.delta_c,self.splits) + phi_u
                phi = Tools.small_2_big(phi_c,un_keys['delta_e'],col_keys['delta_e'],col_keys['splits']) + phi_u
            else:
                phi = phi_c + phi_u
            # Step 4: Calculate next time step
            source = np.einsum('ijk,ik->ij',uncollided.scatter,phi) + uncollided.source + np.einsum('ijk,ik->ij',uncollided.fission,phi)
            phi,psi_next = uncollided.multi_group(psi_last,speed_u,self.geometry,source)
            # Step 5: Update and repeat
            #print('Time Step',t,'Flux',np.sum(phi),'\n===================================')
            
            if self.ptype in ['Stainless','UraniumStainless','StainlessUranium']: # and t == 0: # kill source after first time step
                if t < int(0.2*steps):
                    uncollided.lhs *= 1
                    
                elif t % int(0.1*steps) == 0:
                    # t >= int(0.2*steps) and 
                    print('Reduction by 1/2')
                    uncollided.lhs *= 0.5
                    print('Source ',np.sum(uncollided.lhs))

            psi_last = psi_next.copy(); time_phi.append(phi)

        return phi,time_phi

    def bdf2(self):

        if self.enrich:
            un_attr,un_keys = FixedSource.initialize(self.ptype,self.Gu,self.Nu,T=self.T,dt=self.dt,hybrid=self.Gu,enrich=self.enrich,edges=self.edges)
            uncollided = Uncollided(*un_attr,td='BDF2')
            col_attr,col_keys = FixedSource.initialize(self.ptype,self.Gc,self.Nc,T=self.T,dt=self.dt,hybrid=self.Gu,enrich=self.enrich,edges=self.edges)
            collided = Collided(*col_attr,td='BDF2')
        else:
            un_attr,un_keys = FixedSource.initialize(self.ptype,self.Gu,self.Nu,T=self.T,dt=self.dt,hybrid=self.Gu,edges=self.edges)
            uncollided = Uncollided(*un_attr,td='BDF2')
            col_attr,col_keys = FixedSource.initialize(self.ptype,self.Gc,self.Nc,T=self.T,dt=self.dt,hybrid=self.Gu,edges=self.edges)
            collided = Collided(*col_attr,td='BDF2')
            
        time_phi = []
        speed_u = 1/(un_keys['v']*un_keys['dt']); speed_c = 1/(col_keys['v']*col_keys['dt'])
        phi_c = np.zeros((collided.I,collided.G))

        print('Initial Source ',np.sum(uncollided.lhs))
        steps = int(un_keys['T']/un_keys['dt'])

        # Initialize psi to zero
        psi_n0 = np.zeros((uncollided.I,uncollided.N,uncollided.G))
        psi_n1 = psi_n0.copy()
        for t in range(int(un_keys['T']/un_keys['dt'])): 
            psi_last = 2 * psi_n1 - 0.5 * psi_n0
            # Step 1: Solve Uncollided Equation
            phi_u,_ = uncollided.multi_group(psi_last,speed_u,self.geometry)
            # Step 2: Compute Source for Collided
            source_c = np.einsum('ijk,ik->ij',uncollided.scatter,phi_u) + np.einsum('ijk,ik->ij',uncollided.fission,phi_u) 
            # Resizing
            if self.Gu != self.Gc:
                source_c = Tools.big_2_small(source_c,un_keys['delta_e'],col_keys['delta_e'],col_keys['splits'])
            # Step 3: Solve Collided Equation
            phi_c = collided.multi_group(speed_c,source_c,phi_c,self.geometry)
            # Resize phi_c
            if self.Gu != self.Gc:
                # phi = Tools.small_2_big(phi_c,self.delta_u,self.delta_c,self.splits) + phi_u
                phi = Tools.small_2_big(phi_c,un_keys['delta_e'],col_keys['delta_e'],col_keys['splits']) + phi_u
            else:
                phi = phi_c + phi_u
            # Step 4: Calculate next time step
            source = np.einsum('ijk,ik->ij',uncollided.scatter,phi) + uncollided.source + np.einsum('ijk,ik->ij',uncollided.fission,phi)
            phi,psi_next = uncollided.multi_group(psi_last,speed_u,self.geometry,source)
            # Step 5: Update and repeat
            print('Time Step',t,'Flux',np.sum(phi),'\n===================================')
            
            if self.ptype in ['Stainless','UraniumStainless','StainlessUranium']: # and t == 0: # kill source after first time step
                if t < int(0.2*steps):
                    uncollided.lhs *= 1
                    
                elif t % int(0.1*steps) == 0:
                    # t >= int(0.2*steps) and 
                    print('Reduction by 1/2')
                    uncollided.lhs *= 0.5
                    print('Source ',np.sum(uncollided.lhs))

            # psi_last = psi_next.copy(); 
            time_phi.append(phi)

            psi_n0 = psi_n1.copy()
            psi_n1 = psi_next.copy()

        return phi,time_phi

class Uncollided:
    def __init__(self,G,N,mu,w,total,scatter,fission,source,I,delta,lhs,td='BE'):
        self.G = G; self.N = N; 
        self.mu = mu; self.w = w
        self.total = total
        self.scatter = scatter
        self.fission = fission
        self.source = source
        self.I = I
        self.delta = 1/delta
        self.lhs = lhs
        self.td = td

    def slab(self,psi_last,speed,total_,source_,boundary):
        """ Step 1 of Hybrid
        Arguments:
            Different variables for collided and uncollided except I and inv_delta 
            psi_last: last time step, of size I x N
            speed: 1/(v*dt)   """

        clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cHybrid.so')
        sweep = clibrary.uncollided

        phi = np.zeros((self.I),dtype='float64')

        psi_next = np.zeros(psi_last.shape,dtype='float64')

        weight = self.mu * self.delta
        lhs = ctypes.c_double(boundary)

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

            if self.td == 'BE':
                top_mult = (weight - 0.5 * total_ - 0.5 * speed).astype('float64')
                bottom_mult = (1/(weight + 0.5 * total_ + 0.5 * speed)).astype('float64')
            elif self.td == 'BDF2':
                top_mult = (weight - 0.5 * total_ - 0.75 * speed).astype('float64')
                bottom_mult = (1/(weight + 0.5 * total_ + 0.75 * speed)).astype('float64')

            # top_mult = (weight - 0.5 * total_ - 0.5 * speed).astype('float64')
            # bottom_mult = (1/(weight + 0.5 * total_ + 0.5 * speed)).astype('float64')
            top_ptr = ctypes.c_void_p(top_mult.ctypes.data)
            bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)

            phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                
            sweep(phi_ptr,psi_ptr,rhs_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]),lhs,direction)

            psi_next[:,n] = psi_angle.copy()
        
        return phi,psi_next

    def sphere(self,psi_last,speed,total_,source_,boundary):
        clib = ctypes.cdll.LoadLibrary('./discrete1/data/cHybridSP.so')

        edges = np.cumsum(np.insert(np.ones((self.I))*1/self.delta,0,0))
        SA_plus = Tools.surface_area(edges[1:]).astype('float64')       # Positive surface area
        SAp_ptr = ctypes.c_void_p(SA_plus.ctypes.data)

        SA_minus = Tools.surface_area(edges[:self.I]).astype('float64') # Negative surface area
        SAm_ptr = ctypes.c_void_p(SA_minus.ctypes.data)

        V = Tools.volume(edges[1:],edges[:self.I])                      # Volume and total
        if self.td == 'BE':
            v_total = (V * total_ + speed).astype('float64')
        elif self.td == 'BDF2':
            v_total = (V * total_ + 1.5 * speed).astype('float64')
        v_ptr = ctypes.c_void_p(v_total.ctypes.data)

        psi_next = np.zeros(psi_last.shape,dtype='float64')
        psi_centers = np.zeros((self.N),dtype='float64')

        angular = np.zeros((self.I),dtype='float64')
        an_ptr = ctypes.c_void_p(angular.ctypes.data)

        phi = np.zeros((self.I),dtype='float64')
        for n in range(self.N):
            if n == 0:
                alpha_minus = ctypes.c_double(0.)
                psi_nhalf = (Tools.half_angle(0,total_,1/self.delta, source_)).astype('float64')
            if n == self.N - 1:
                alpha_plus = ctypes.c_double(0.)
            else:
                alpha_plus = ctypes.c_double(alpha_minus - self.mu[n] * self.w[n])

            # psi_ihalf = ctypes.c_double(min(0.,psi_centers[N-n-1],key=abs))
            if self.mu[n] > 0:
                psi_ihalf = ctypes.c_double(psi_centers[self.N-n-1])
            elif self.mu[n] < 0:
                psi_ihalf = ctypes.c_double(boundary)

            Q = (V * (source_) + psi_last[:,n] * speed).astype('float64') #+ psi_last[:,n] * speed
            q_ptr = ctypes.c_void_p(Q.ctypes.data)

            psi_ptr = ctypes.c_void_p(psi_nhalf.ctypes.data)
            phi_ptr = ctypes.c_void_p(phi.ctypes.data)

            clib.sweep(an_ptr,phi_ptr,psi_ptr,q_ptr,v_ptr,SAp_ptr,SAm_ptr,ctypes.c_double(self.w[n]),ctypes.c_double(self.mu[n]),alpha_plus,alpha_minus,psi_ihalf)
            # Update angular center corrections
            psi_centers[n] = angular[0]
            psi_next[:,n] = angular.copy()
            # Update angular difference coefficients
            alpha_minus = alpha_plus
                
        return phi,psi_next

    def multi_group(self,psi_last,speed,geometry='slab',source=None):
        # G is Gu
        phi_old = np.zeros((self.I,self.G))
        psi_next = np.zeros(psi_last.shape)

        geo = getattr(Uncollided,geometry)  # Get the specific sweep

        if source is None:
            current = self.source.copy()
        else:
            current = source.copy()

        tol = 1e-12; MAX_ITS = 100
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)
            for g in range(self.G):
                phi[:,g],psi_next[:,:,g] = geo(self,psi_last[:,:,g],speed[g],self.total[:,g],current[:,g],self.lhs[g])
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            if np.isnan(change) or np.isinf(change):
                change = 0.
            # print('Uncollided Change is',change,'\n===================================')
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 

            phi_old = phi.copy()

        return phi,psi_next


class Collided:
    def __init__(self,G,N,mu,w,total,scatter,fission,source,I,delta,boundary,td='BE'):
        self.G = G; self.N = N; 
        self.mu = mu; self.w = w
        self.total = total
        self.scatter = scatter
        self.fission = fission
        # self.source = source
        self.I = I
        self.delta = 1/delta
        # self.boundary = boundary
        self.td = td

    def slab(self,speed,total_,scatter_,source_,guess_):

        clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cHybrid.so')
        sweep = clibrary.collided

        source_ = source_.astype('float64')
        source_ptr = ctypes.c_void_p(source_.ctypes.data)
        
        phi_old = guess_.copy()

        tol = 1e-8; MAX_ITS = 100
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros((self.I),dtype='float64')
            for n in range(self.N):
                # Determine the direction
                direction = ctypes.c_int(int(np.sign(self.mu[n])))
                weight = np.sign(self.mu[n]) * self.mu[n] * self.delta

                if self.td == 'BE':
                    top_mult = (weight - 0.5 * total_ - 0.5 * speed).astype('float64')
                    bottom_mult = (1/(weight + 0.5 * total_ + 0.5 * speed)).astype('float64')
                elif self.td == 'BDF2':
                    top_mult = (weight - 0.5 * total_ - 0.75 * speed).astype('float64')
                    bottom_mult = (1/(weight + 0.5 * total_ + 0.75 * speed)).astype('float64')
                # top_mult = (weight - 0.5 * total_ - 0.5 * speed).astype('float64')
                # bottom_mult = (1/(weight + 0.5 * total_ + 0.5 * speed)).astype('float64') 
                top_ptr = ctypes.c_void_p(top_mult.ctypes.data)
                bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)

                temp_scat = (scatter_ * phi_old).astype('float64')
                ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)

                phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                    
                sweep(phi_ptr,ts_ptr,source_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]),direction)

            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            if np.isnan(change) or np.isinf(change):
                change = 0.
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi

    def sphere(self,speed,total_,scatter_,source_,guess_):
        clib = ctypes.cdll.LoadLibrary('./discrete1/data/cHybridSP.so')

        edges = np.cumsum(np.insert(np.ones((self.I))*1/self.delta,0,0))
        SA_plus = Tools.surface_area(edges[1:]).astype('float64')       # Positive surface area
        SAp_ptr = ctypes.c_void_p(SA_plus.ctypes.data)

        SA_minus = Tools.surface_area(edges[:self.I]).astype('float64') # Negative surface area
        SAm_ptr = ctypes.c_void_p(SA_minus.ctypes.data)

        V = Tools.volume(edges[1:],edges[:self.I])                      # Volume and total
        if self.td == 'BE':
            v_total = (V * total_ + speed).astype('float64')
        elif self.td == 'BDF2':
            v_total = (V * total_ + 1.5 * speed).astype('float64')
        v_ptr = ctypes.c_void_p(v_total.ctypes.data)

        phi_old = guess_.copy()
        # psi_next = np.zeros(psi_last.shape,dtype='float64')
        psi_centers = np.zeros((self.N),dtype='float64')

        tol=1e-12; MAX_ITS=100
        converged = 0; count = 1; 
        while not (converged):
            angular = np.zeros((self.I),dtype='float64')
            an_ptr = ctypes.c_void_p(angular.ctypes.data)

            phi = np.zeros((self.I),dtype='float64')
            # psi_nhalf = (Tools.half_angle(0,total_,1/self.delta, source_ + scatter_ * phi_old)).astype('float64')
            for n in range(self.N):
                if n == 0:
                    alpha_minus = ctypes.c_double(0.)
                    psi_nhalf = (Tools.half_angle(0,total_,1/self.delta, source_ + scatter_ * phi_old)).astype('float64')
                if n == self.N - 1:
                    alpha_plus = ctypes.c_double(0.)
                else:
                    alpha_plus = ctypes.c_double(alpha_minus - self.mu[n] * self.w[n])

                # psi_ihalf = ctypes.c_double(min(0.,psi_centers[N-n-1],key=abs))
                psi_ihalf = ctypes.c_double(0.)
                # if self.mu[n] > 0:
                #     psi_ihalf = ctypes.c_double(psi_centers[self.N-n-1])
                # elif self.mu[n] < 0:
                #     psi_ihalf = ctypes.c_double(boundary)

                Q = (V * (source_ + scatter_ * phi_old)).astype('float64') #+ psi_last[:,n] * speed
                q_ptr = ctypes.c_void_p(Q.ctypes.data)

                psi_ptr = ctypes.c_void_p(psi_nhalf.ctypes.data)
                phi_ptr = ctypes.c_void_p(phi.ctypes.data)

                clib.sweep(an_ptr,phi_ptr,psi_ptr,q_ptr,v_ptr,SAp_ptr,SAm_ptr,ctypes.c_double(self.w[n]),ctypes.c_double(self.mu[n]),alpha_plus,alpha_minus,psi_ihalf)
                # Update angular center corrections
                psi_centers[n] = angular[0]
                # Update angular difference coefficients
                alpha_minus = alpha_plus
                
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()

        return phi

    def update_q(xs,phi,start,stop,g):
        return np.sum(xs[:,g,start:stop]*phi[:,start:stop],axis=1)

    def multi_group(self,speed,source,guess,geometry='slab'):
        
        assert(source.shape[1] == self.G), 'Wrong Number of Groups'

        phi_old = guess.copy()

        geo = getattr(Collided,geometry)  # Get the specific sweep
        
        tol = 1e-12; MAX_ITS = 100
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)
            for g in range(self.G):
                q_tilde = source[:,g] + Collided.update_q(self.scatter,phi_old,g+1,self.G,g) + Collided.update_q(self.fission,phi_old,g+1,self.G,g)
                if g != 0:
                    q_tilde += Collided.update_q(self.scatter,phi,0,g,g) + Collided.update_q(self.fission,phi,0,g,g)
                phi[:,g] = geo(self,speed[g],self.total[:,g],self.scatter[:,g,g]+self.fission[:,g,g],q_tilde,phi_old[:,g])
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            if np.isnan(change) or np.isinf(change):
                change = 0.5
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

        new_grid = np.ones((small)) * int(big/small)
        new_grid[np.linspace(0,small-1,big % small,dtype=int)] += 1

        inds = np.cumsum(np.insert(new_grid,0,0),dtype=int)

        splits = [slice(ii,jj) for ii,jj in zip(inds[:small],inds[1:])]
        return splits

    def small_2_big(mult_c,delta_u,delta_c,splits):

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

        size = (mult_u.shape[0],len(delta_c))
        mult_c = np.zeros(size)

        for count,index in enumerate(splits):
            mult_c[:,count] = np.sum(mult_u[:,index],axis=1) 

        return mult_c

    def surface_area(rho):
        return 4 * np.pi * rho**2

    def volume(plus,minus):
        return 4 * np.pi / 3 * (plus**3 - minus**3)

    def half_angle(psi_plus,total,delta,source):
        """ This is for finding the half angle (N = 1/2) at cell i """
        return (2 * psi_plus + delta * source ) / (2 + total * delta)
        
