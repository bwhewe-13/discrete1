"""
Running Hybrid Problems (time dependent)
"""

from .fixed import FixedSource, ControlRod

import numpy as np
import ctypes
from scipy.special import erfc

class Hybrid:
    def __init__(self,ptype,G,N):
        """ G and N are lists of [uncollided,collided]  """
        self.ptype = ptype
        self.geometry = 'slab'
        self.Gu,self.Gc = G if type(G) is list else [G,G]
        self.Nu,self.Nc = N if type(N) is list else [N,N]

    @classmethod
    def run(cls,ptype,G,N,T,dt,**kwargs):
        prob = cls(ptype,G,N)
        prob.T = T; prob.dt = dt
        prob.edges = kwargs['edges'] if 'edges' in kwargs else None
        prob.enrich = kwargs['enrich'] if 'enrich' in kwargs else None
        prob.geometry = kwargs['geometry'] if 'geometry' in kwargs else 'slab'
        if 'td' in kwargs:
            prob.td = kwargs['td']
            if prob.td == 'BE':
                return prob.backward_euler()
            elif prob.td == 'BDF2':
                return prob.bdf2()
        return prob.backward_euler()

    def backward_euler(self):
        # Set up Problem
        un_attr,un_keys = FixedSource.initialize(self.ptype,self.Gu,self.Nu,T=self.T,dt=self.dt,hybrid=self.Gu,enrich=self.enrich,edges=self.edges)
        uncollided = Uncollided(*un_attr)
        col_attr,col_keys = FixedSource.initialize(self.ptype,self.Gc,self.Nc,T=self.T,dt=self.dt,hybrid=self.Gu,enrich=self.enrich,edges=self.edges)
        collided = Collided(*col_attr)
        # Initialize Speeds
        speed_u = 1/(un_keys['v']*un_keys['dt'])
        speed_c = 1/(col_keys['v']*col_keys['dt'])
        # Initialize collided scalar flux
        phi_c = np.zeros((collided.I,collided.G)); time_phi = []
        psi_last = np.zeros((uncollided.I,uncollided.N,uncollided.G))
        if self.ptype in ['ControlRod']:
            # psi_last = np.tile(np.expand_dims(np.load('discrete1/data/initial_rod.npy'),axis=1),(1,uncollided.N,1))
            psi_last = np.tile(np.expand_dims(np.load('mydata/control_rod_critical/carbon_g87_phi_20.npy'),axis=1),(1,uncollided.N,1))
        # For calculating the number of time steps (computer rounding error)
        steps = int(np.round(un_keys['T']/un_keys['dt'],5))
        # Determining source for problem
        if self.ptype in ['Stainless','UraniumStainless','StainlessUranium']:
            # full_lhs = Tools.discontinuous(uncollided.lhs,steps)
            full_lhs = Tools.continuous(uncollided.lhs,steps)
        else:
            full_lhs = Tools.stagnant(uncollided.lhs,steps)
        for t in range(steps):
            if self.ptype in ['ControlRod']:
                # The change of carbon --> stainless in problem
                switch = min(max(np.round(1 - 10**-(len(str(steps))-1)*10**(len(str(int(un_keys['T']/1E-6)))-1) * t,2),0),1)
                print('Switch {} Step {}'.format(switch,t))
                uncollided.total,uncollided.scatter,uncollided.fission = ControlRod.xs_update(uncollided.G,enrich=0.20,switch=switch)
                collided.total,collided.scatter,collided.fission = ControlRod.xs_update(collided.G,enrich=0.20,switch=switch)
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
            print('Time Step',t,'Flux',np.sum(phi),'\n===================================')
            # Step 5: Update and repeat
            psi_last = psi_next.copy()
            time_phi.append(phi)
            uncollided.lhs = full_lhs[t].copy()
        return phi,time_phi

    def bdf2(self):
        # Set up Problem
        un_attr,un_keys = FixedSource.initialize(self.ptype,self.Gu,self.Nu,T=self.T,dt=self.dt,hybrid=self.Gu,enrich=self.enrich,edges=self.edges)
        uncollided = Uncollided(*un_attr,td='BDF2')
        col_attr,col_keys = FixedSource.initialize(self.ptype,self.Gc,self.Nc,T=self.T,dt=self.dt,hybrid=self.Gu,enrich=self.enrich,edges=self.edges)
        collided = Collided(*col_attr,td='BDF2')        
        # Initialize speed
        speed_u = 1/(un_keys['v']*un_keys['dt'])
        speed_c = 1/(col_keys['v']*col_keys['dt'])
        # Initialize collided scalar flux
        phi_c = np.zeros((collided.I,collided.G)); time_phi = []
        psi_n0 = np.zeros((uncollided.I,uncollided.N,uncollided.G)); psi_n1 = psi_n0.copy()        
        # Initialize ControlRod problem differently
        if self.ptype in ['ControlRod']:
            # psi_n1 = np.tile(np.expand_dims(np.load('discrete1/data/initial_rod.npy'),axis=1),(1,uncollided.N,1))
            psi_last = np.tile(np.expand_dims(np.load('mydata/control_rod_critical/carbon_g87_phi_20.npy'),axis=1),(1,uncollided.N,1))
            collided.td = 'BE'; uncollided.td = 'BE'
        # For calculating the number of time steps (computer rounding error)
        steps = int(np.round(un_keys['T']/un_keys['dt'],5))
        # Determining source for problem
        if self.ptype in ['Stainless','UraniumStainless','StainlessUranium']:
            # full_lhs = Tools.discontinuous(uncollided.lhs,steps)
            full_lhs = Tools.continuous(uncollided.lhs,steps)
        else:
            full_lhs = Tools.stagnant(uncollided.lhs,steps)
        for t in range(steps):
            if self.ptype in ['ControlRod']:
                # The change of carbon --> stainless in problem
                switch = min(max(np.round(1 - 10**-(len(str(steps))-1)*10**(len(str(int(un_keys['T']/1E-6)))-1) * t,2),0),1)
                print('Switch {} Step {}'.format(switch,t))
                uncollided.total,uncollided.scatter,uncollided.fission = ControlRod.xs_update(uncollided.G,enrich=0.20,switch=switch)
                collided.total,collided.scatter,collided.fission = ControlRod.xs_update(collided.G,enrich=0.20,switch=witch)
            # Backward Euler for first step, BDF2 for rest
            psi_last = psi_n1.copy() if t == 0 else 2 * psi_n1 - 0.5 * psi_n0
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
            psi_n0 = psi_n1.copy()
            psi_n1 = psi_next.copy()
            time_phi.append(phi)
            uncollided.lhs = full_lhs[t].copy()
            # change to BDF2 time steps
            collided.td = 'BDF2'; uncollided.td = 'BDF2'
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

        mu_minus = -1
        angular = np.zeros((self.I),dtype='float64')
        an_ptr = ctypes.c_void_p(angular.ctypes.data)
        phi = np.zeros((self.I),dtype='float64')
        for n in range(self.N):
            mu_plus = mu_minus + 2 * self.w[n]
            tau = (self.mu[n] - mu_minus) / (mu_plus - mu_minus)
            if n == 0:
                alpha_minus = ctypes.c_double(0.)
                psi_nhalf = (Tools.half_angle(boundary,total_,1/self.delta, source_)).astype('float64')
            if n == self.N - 1:
                alpha_plus = ctypes.c_double(0.)
            else:
                alpha_plus = ctypes.c_double(alpha_minus - self.mu[n] * self.w[n])
            # psi_ihalf = ctypes.c_double(min(0.,psi_centers[N-n-1],key=abs))
            if self.mu[n] > 0:
                # psi_ihalf = ctypes.c_double(psi_centers[self.N-n-1])
                psi_ihalf = ctypes.c_double(psi_nhalf[0])
            elif self.mu[n] < 0:
                psi_ihalf = ctypes.c_double(boundary)
            Q = (V * (source_) + psi_last[:,n] * speed).astype('float64') #+ psi_last[:,n] * speed
            q_ptr = ctypes.c_void_p(Q.ctypes.data)
            psi_ptr = ctypes.c_void_p(psi_nhalf.ctypes.data)
            phi_ptr = ctypes.c_void_p(phi.ctypes.data)
            clib.sweep(an_ptr,phi_ptr,psi_ptr,q_ptr,v_ptr,SAp_ptr,SAm_ptr,ctypes.c_double(self.w[n]),ctypes.c_double(self.mu[n]),alpha_plus,alpha_minus,psi_ihalf,ctypes.c_double(tau))
            # Update angular center corrections
            psi_centers[n] = angular[0]
            psi_next[:,n] = angular.copy()
            # Update angular difference coefficients
            alpha_minus = alpha_plus
            mu_minus = mu_plus
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
            mu_minus = -1
            angular = np.zeros((self.I),dtype='float64')
            an_ptr = ctypes.c_void_p(angular.ctypes.data)

            phi = np.zeros((self.I),dtype='float64')
            for n in range(self.N):
                mu_plus = mu_minus + 2 * self.w[n]
                tau = (self.mu[n] - mu_minus) / (mu_plus - mu_minus)
                if n == 0:
                    alpha_minus = ctypes.c_double(0.)
                    psi_nhalf = (Tools.half_angle(0,total_,1/self.delta, source_ + scatter_ * phi_old)).astype('float64')
                if n == self.N - 1:
                    alpha_plus = ctypes.c_double(0.)
                else:
                    alpha_plus = ctypes.c_double(alpha_minus - self.mu[n] * self.w[n])
                if self.mu[n] > 0:
                    # psi_ihalf = ctypes.c_double(psi_centers[self.N-n-1])
                    psi_ihalf = ctypes.c_double(psi_nhalf[0])
                elif self.mu[n] < 0:
                    psi_ihalf = ctypes.c_double(0.)

                Q = (V * (source_ + scatter_ * phi_old)).astype('float64') #+ psi_last[:,n] * speed
                q_ptr = ctypes.c_void_p(Q.ctypes.data)
                psi_ptr = ctypes.c_void_p(psi_nhalf.ctypes.data)
                phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                clib.sweep(an_ptr,phi_ptr,psi_ptr,q_ptr,v_ptr,SAp_ptr,SAm_ptr,ctypes.c_double(self.w[n]),ctypes.c_double(self.mu[n]),alpha_plus,alpha_minus,psi_ihalf,ctypes.c_double(tau))
                # Update angular center corrections
                psi_centers[n] = angular[0]
                # Update angular difference coefficients
                alpha_minus = alpha_plus
                mu_minus = mu_plus
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
        # return (2 * psi_plus + delta * source ) / (2 + total * delta)
        psi_nhalf = np.zeros((len(total)))
        for ii in range(len(total)-1,-1,-1):
            psi_nhalf[ii] = (2 * psi_plus + delta * source[ii] ) / (2 + total[ii] * delta)
            psi_plus = 2 * psi_nhalf[ii] - psi_plus
        return psi_nhalf
        
    def stagnant(source,steps):
        return np.tile(source,(steps,1))

    def continuous(source,steps):
        func = lambda a,b: list(erfc(np.arange(1,4))*a+b)
        full = np.zeros((steps,len(source)))
        group = np.argwhere(source != 0)[0,0]
        source = source[group]
        for t in range(steps):
            if t < int(0.2*steps):
                source *= 1
                full[t,group] = source
            elif t % int(0.1*steps) == 0:
                temp = t
                full[t:t+3,group] = func(source,0.5*source)
                source *= 0.5
            elif t in np.arange(temp+1,temp+3):
                continue
            else:
                full[t,group] = source
        return full

    def discontinuous(source,steps):
        full = np.zeros((steps,len(source)))
        group = np.argwhere(source != 0)[0,0]
        source = source[group]
        for t in range(steps):
            if t < int(0.2*steps):
                source *= 1
            elif t % int(0.1*steps) == 0:
                source *= 0.5
            full[t,group] = source
        return full
