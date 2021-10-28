"""
Running Multigroup Source Problems
"""

from .fixed import FixedSource,ControlRod

import numpy as np
import ctypes
from scipy.special import erfc

import time

class Source:
    # Keyword Arguments allowed currently
    __allowed = ("T","dt","v","boundary","geometry","td") # ,"hybrid","enrich","track")

    def __init__(self,G,N,mu,w,total,scatter,fission,source,I,delta,lhs,**kwargs): 
        """ Deals with Source multigroup problems (time dependent, steady state,
        reflected and vacuum boundaries)
        Attributes:
            G: Number of energy groups, int
            N: Number of discrete angles, int
            mu: Angles from Legendre Polynomials, numpy array of size (N,)
            w: Normalized weights for mu, numpy array of size (N,)
            total: total cross section, numpy array of size (I x G)
            scatter: scatter cross section, numpy array of size (I x G x G)
            fission: fission cross section, numpy array of size (I x G x G)
            source: source, numpy array of size (I x G)
            I: number of spatial cells, int
            delta: width of each spatial cell, int
            boundary: Sources entering from LHS, numpy array of size (1 x G)
        kwargs:
            T: Length of the time period (for time dependent problems), float
            dt: Width of the time step (for time dependent problems), float
            v: Speed of the neutrons in cm/s, list of length G
            boundary: str (default vacuum), determine RHS of problem
                options: 'vacuum', 'reflected'
            enrich: enrichment percentage of U235 in uranium problems, float
            track: bool (default False), if track flux change with iteration
            geometry: str (default slab), determines coordinates of sweep ('slab' or 'sphere')
            td (Time Discretization): str (default BE), which time discretization to use
        """ 
        # Attributes
        self.G = G; self.N = N
        self.mu = mu; self.w = w
        self.total = total
        self.scatter = scatter
        self.fission = fission
        self.source = source
        self.I = I
        self.delta = 1/delta
        self.lhs = lhs
        # kwargs
        self.T = None; self.dt = None; self.v = None
        self.boundary = 'vacuum'; self.problem = None
        self.geometry = 'slab'; self.td = 'BE'
        #self.enrich = None; self.time = False
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, available: boundary, time, geometry" 
            setattr(self, key, value)

    @classmethod
    def run(cls,ptype,G,N,**kwargs):
        attributes,keywords = FixedSource.initialize(ptype,G,N,**kwargs)
        problem = cls(*attributes,**keywords)
        problem.problem = ptype
        if 'boundary' in kwargs:
            boundary = kwargs['boundary']
            # Currently cannot use reflected boundary with TD problems
            if 'T' in kwargs and boundary == 'reflected':
                print('Using Vacuum Boundaries, cannot use Time Dependency \
                    with Reflected Conditions')
                boundary = 'vacuum'
            problem.boundary = boundary
        if 'geometry' in kwargs:
            geometry = kwargs['geometry']
            problem.geometry = geometry
        if 'T' in kwargs and 'td' not in kwargs:
            kwargs['td'] = 'BE'
        if 'td' in kwargs:
            problem.td = kwargs['td']
            if problem.td == 'BE':
                return problem.backward_euler()
            elif problem.td == 'BDF2':
                return problem.bdf2()
        return problem.multi_group()
               
    def slab(self,total_,scatter_,source_,boundary,guess):
        """ Arguments:
            total: I x 1 vector of the total cross section for each spatial cell
            scatter: I x L+1 array for the scattering of the spatial cell by moment
            external: I array for the external sources
            guess: Initial guess of the scalar flux for a specific energy group (I x L+1)
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
        Returns:
            phi: a I array  """
        clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cSource.so')
        sweep = clibrary.vacuum
        if self.boundary == 'reflected':
            # print('\nReflected\n')
            sweep = clibrary.reflected

        phi_old = guess.copy()
        
        source_ = source_.astype('float64')
        source_ptr = ctypes.c_void_p(source_.ctypes.data)

        lhs = ctypes.c_double(boundary)

        tol = 1e-12; MAX_ITS = 100
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros((self.I),dtype='float64')
            for n in range(self.N):
                direction = ctypes.c_int(int(np.sign(self.mu[n])))
                weight = np.sign(self.mu[n]) * self.mu[n] * self.delta

                top_mult = (weight - 0.5 * total_).astype('float64')
                top_ptr = ctypes.c_void_p(top_mult.ctypes.data)

                bottom_mult = (1/(weight + 0.5 * total_)).astype('float64')
                bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)

                temp_scat = (scatter_ * phi_old).astype('float64')
                ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)
                
                phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                
                sweep(phi_ptr,ts_ptr,source_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]),lhs,direction)

            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            if np.isnan(change) or np.isinf(change):
                change = 0.
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()

        return phi

    def sphere(self,total_,scatter_,source_,boundary,guess):
        """ Arguments:
            total_: I x 1 vector of the total cross section for each spatial cell
            scatter_: I x L+1 array for the scattering of the spatial cell by moment
            source_: I array for the external sources
            guess: Initial guess of the scalar flux for a specific energy group (I x L+1)
        Returns:
            phi: a I array  """
        clib = ctypes.cdll.LoadLibrary('./discrete1/data/cSourceSP.so')

        edges = np.cumsum(np.insert(np.ones((self.I))*1/self.delta,0,0))
        SA_plus = Source.surface_area(edges[1:]).astype('float64')       # Positive surface area
        SAp_ptr = ctypes.c_void_p(SA_plus.ctypes.data)

        SA_minus = Source.surface_area(edges[:self.I]).astype('float64') # Negative surface area
        SAm_ptr = ctypes.c_void_p(SA_minus.ctypes.data)

        V = Source.volume(edges[1:],edges[:self.I])                      # Volume and total
        v_total = (V * total_).astype('float64')
        v_ptr = ctypes.c_void_p(v_total.ctypes.data)

        phi_old = guess.copy()
        psi_centers = np.zeros((self.N),dtype='float64')

        tol=1e-12; MAX_ITS=100
        converged = 0; count = 1; 
        while not (converged):
            mu_minus = -1

            angular = np.zeros((self.I),dtype='float64')
            an_ptr = ctypes.c_void_p(angular.ctypes.data)

            phi = np.zeros((self.I),dtype='float64')
            # psi_nhalf = (Source.half_angle(0,total_,1/self.delta, source_ + scatter_ * phi_old)).astype('float64')
            for n in range(self.N):
                mu_plus = mu_minus + 2 * self.w[n]
                tau = (self.mu[n] - mu_minus) / (mu_plus - mu_minus)
                if n == 0:
                    alpha_minus = ctypes.c_double(0.)
                    psi_nhalf = (Source.half_angle(0,total_,1/self.delta, source_ + scatter_ * phi_old)).astype('float64')
                if n == self.N - 1:
                    alpha_plus = ctypes.c_double(0.)
                else:
                    alpha_plus = ctypes.c_double(alpha_minus - self.mu[n] * self.w[n])

                if self.mu[n] > 0:
                    # psi_ihalf = ctypes.c_double(psi_centers[self.N-n-1])
                    psi_ihalf = ctypes.c_double(psi_nhalf[0])
                elif self.mu[n] < 0:
                    psi_ihalf = ctypes.c_double(boundary)

                Q = (V * (source_ + scatter_ * phi_old)).astype('float64')
                q_ptr = ctypes.c_void_p(Q.ctypes.data)

                psi_ptr = ctypes.c_void_p(psi_nhalf.ctypes.data)
                phi_ptr = ctypes.c_void_p(phi.ctypes.data)

                clib.sweep(an_ptr,phi_ptr,psi_ptr,q_ptr,v_ptr,SAp_ptr,SAm_ptr,\
                    ctypes.c_double(self.w[n]),ctypes.c_double(self.mu[n]),alpha_plus,\
                    alpha_minus,psi_ihalf,ctypes.c_double(tau))
                # Update angular center corrections
                # psi_centers[n] = angular[0]
                # Update angular difference coefficients
                alpha_minus = alpha_plus
                mu_minus = mu_plus
                
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()

        return phi

    def sphere_time(self,total_,scatter_,source_,boundary,guess,psi_last,speed):
        """ Arguments:
            total_: I x 1 vector of the total cross section for each spatial cell
            scatter_: I x L+1 array for the scattering of the spatial cell by moment
            source_: I array for the external sources
            guess: Initial guess of the scalar flux for a specific energy group (I x L+1)
        Returns:
            phi: a I array  """
        clib = ctypes.cdll.LoadLibrary('./discrete1/data/cSourceSP.so')

        edges = np.cumsum(np.insert(np.ones((self.I))*1/self.delta,0,0))
        SA_plus = Source.surface_area(edges[1:]).astype('float64')       # Positive surface area
        SAp_ptr = ctypes.c_void_p(SA_plus.ctypes.data)

        SA_minus = Source.surface_area(edges[:self.I]).astype('float64') # Negative surface area
        SAm_ptr = ctypes.c_void_p(SA_minus.ctypes.data)

        V = Source.volume(edges[1:],edges[:self.I])                      # Volume and total
        if self.td == 'BE':
            v_total = (V * total_ + speed).astype('float64')
        elif self.td == 'BDF2':
            v_total = (V * total_ + 1.5 * speed).astype('float64')

        v_ptr = ctypes.c_void_p(v_total.ctypes.data)

        phi_old = guess.copy()
        psi_next = np.zeros(psi_last.shape,dtype='float64')
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
                    psi_nhalf = (Source.half_angle(boundary,total_,1/self.delta, source_ + scatter_ * phi_old)).astype('float64')
                if n == self.N - 1:
                    alpha_plus = ctypes.c_double(0.)
                else:
                    alpha_plus = ctypes.c_double(alpha_minus - self.mu[n] * self.w[n])

                if self.mu[n] > 0:
                    # psi_ihalf = ctypes.c_double(psi_centers[self.N-n-1])
                    psi_ihalf = ctypes.c_double(psi_nhalf[0])
                elif self.mu[n] < 0:
                    psi_ihalf = ctypes.c_double(boundary)

                Q = (V * (source_ + scatter_ * phi_old) + psi_last[:,n] * speed).astype('float64') #+ psi_last[:,n] * speed
                q_ptr = ctypes.c_void_p(Q.ctypes.data)

                psi_ptr = ctypes.c_void_p(psi_nhalf.ctypes.data)
                phi_ptr = ctypes.c_void_p(phi.ctypes.data)

                clib.sweep(an_ptr,phi_ptr,psi_ptr,q_ptr,v_ptr,SAp_ptr,SAm_ptr,\
                    ctypes.c_double(self.w[n]),ctypes.c_double(self.mu[n]),alpha_plus,\
                    alpha_minus,psi_ihalf,ctypes.c_double(tau))
                # Update angular center corrections
                psi_centers[n] = angular[0]
                psi_next[:,n] = angular.copy()
                # Update angular difference coefficients
                alpha_minus = alpha_plus
                mu_minus = mu_plus
                
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()

        return phi,psi_next

    def slab_time(self,total_,scatter_,source_,boundary,guess,psi_last,speed):
        clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cSource.so')
        sweep = clibrary.time_vacuum

        phi_old = guess.copy().astype('float64')
        # Initialize new angular flux
        psi_next = np.zeros(psi_last.shape,dtype='float64')

        lhs = ctypes.c_double(boundary) 

        tol = 1e-8; MAX_ITS = 100
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros((self.I),dtype='float64')
            for n in range(self.N):
                # Determining the direction
                direction = ctypes.c_int(int(np.sign(self.mu[n])))
                weight = np.sign(self.mu[n]) * self.mu[n] * self.delta
                # Collecting Angle
                psi_angle = np.zeros((self.I),dtype='float64')
                psi_ptr = ctypes.c_void_p(psi_angle.ctypes.data)
                # Source Term
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

                temp_scat = (scatter_ * phi_old).astype('float64')
                ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)
                
                phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                
                sweep(phi_ptr,psi_ptr,ts_ptr,rhs_ptr,top_ptr,bot_ptr,\
                    ctypes.c_double(self.w[n]),lhs,direction)

                psi_next[:,n] = psi_angle.copy()

            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            if np.isnan(change) or np.isinf(change):
                change = 0.
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()

        return phi,psi_next,count

    def update_q(xs,phi,start,stop,g):
        return np.sum(xs[:,g,start:stop]*phi[:,start:stop],axis=1)

    def multi_group(self,**kwargs):
        """ Run multi group steady state problem
        Returns:
            phi: scalar flux, numpy array of size (I x G) """
        phi_old = np.zeros((self.I,self.G))
        geo = getattr(Source,self.geometry)  # Get the specific sweep
        if self.T:
            geo = getattr(Source,'{}_time'.format(self.geometry))  # Get the specific sweep
            psi_last = kwargs['psi_last'].copy()
            psi_next = np.zeros(psi_last.shape)
            phi_old = kwargs['guess'].copy()

        tol = 1e-12; MAX_ITS = 100
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)
            for g in range(self.G):
                q_tilde = self.source[:,g] + Source.update_q(self.scatter,phi_old,g+1,self.G,g) \
                    + Source.update_q(self.fission,phi_old,g+1,self.G,g)
                if g != 0:
                    q_tilde += Source.update_q(self.scatter,phi_old,0,g,g) + Source.update_q(self.fission,phi,0,g,g)
                if self.T:
                    phi[:,g],psi_next[:,:,g],inner_count[g] = geo(self,self.total[:,g],\
                        self.scatter[:,g,g]+self.fission[:,g,g],q_tilde,self.lhs[g],\
                        phi_old[:,g],psi_last[:,:,g],self.speed[g])
                else:
                    phi[:,g] = geo(self,self.total[:,g],self.scatter[:,g,g]+self.fission[:,g,g],q_tilde,self.lhs[g],phi_old[:,g])

            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            if np.isnan(change) or np.isinf(change):
                change = 0.5
            # file = 'testdata/slab_uranium_stainless_ts0100_be/source_multigroup_ts'
            # np.save(file+'{}_{}'.format(str(kwargs['ts']).zfill(4),str(count).zfill(3)),tracking_data)
            # if self.T:
            #     print('Count',count,'Change',change,'\n===================================')
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 
            phi_old = phi.copy()
        if self.T:
            return phi,psi_next
        return phi

    def backward_euler(self):
        # Initialize flux and list of fluxes
        phi_old = np.zeros((self.I,self.G)); time_phi = []
        # Initialize angular flux
        psi_last = np.zeros((self.I,self.N,self.G))
        # Initialize Speed
        self.speed = 1/(self.v*self.dt)
        if self.problem in ['ControlRod']:
            psi_last = np.tile(np.expand_dims(np.load('mydata/control_rod_critical/carbon_g87_phi_15.npy'),axis=1),(1,self.N,1))

        # For calculating the number of time steps (computer rounding error)
        steps = int(np.round(self.T/self.dt,5))
        # Determine source for problem
        if self.problem in ['Stainless','UraniumStainless','StainlessUranium']:
            full_lhs = Source.continuous(self.lhs,steps)
        else:
            full_lhs = Source.stagnant(self.lhs,steps)
        
        for t in range(steps):
            if self.problem in ['ControlRod']:
                # The change of carbon --> stainless in problem
                switch = min(max(np.round(1 - 10**-(len(str(steps))-1)*10**(len(str(int(self.T/1E-6)))-1) * t,2),0),1)
                print('Switch {} Step {}'.format(switch,t))
                self.total,self.scatter,self.fission = ControlRod.xs_update(self.G,enrich=0.15,switch=switch)
            # Run the multigroup problem
            phi,psi_next = Source.multi_group(self,psi_last=psi_last,guess=phi_old,ts=t)
            print('Time Step',t,'Flux',np.sum(phi),'\n===================================')
            # Update scalar/angular flux
            psi_last = psi_next.copy()
            time_phi.append(phi)
            phi_old = phi.copy()
            # Update source
            self.lhs = full_lhs[t].copy()
        return phi,time_phi

    def bdf2(self):
        # Initialize flux and list of fluxes
        phi_old = np.zeros((self.I,self.G)); time_phi = []
        # Initialize angular flux
        psi_n0 = np.zeros((self.I,self.N,self.G)); psi_n1 = psi_n0.copy()
        # Initialize speed
        self.speed = 1/(self.v*self.dt); 
        # Initialize ControlRod problem differently
        if self.problem in ['ControlRod']:
            psi_n1 = np.tile(np.expand_dims(np.load('discrete1/data/initial_rod.npy'),axis=1),(1,self.N,1))
            self.td = 'BE' # Initial step is BE
        # For calculating the number of time steps (computer rounding error)
        steps = int(np.round(self.T/self.dt,5))
        # Determine source for problem
        if self.problem in ['Stainless','UraniumStainless','StainlessUranium']:
            full_lhs = Source.continuous(self.lhs,steps)
        else:
            full_lhs = Source.stagnant(self.lhs,steps)
        for t in range(steps):
            if self.problem in ['ControlRod']:
                # The change of carbon --> stainless in problem
                switch = min(max(np.round(1 - 10**-(len(str(steps))-1)*10**(len(str(int(self.T/1E-6)))-1) * t,2),0),1)
                print('Switch {} Step {}'.format(switch,t))
                self.total,self.scatter,self.fission = ControlRod.xs_update(self.G,enrich=0.20,switch=switch)

            # Backward Euler for first step, BDF2 for rest
            psi_last = psi_n1.copy() if t == 0 else 2 * psi_n1 - 0.5 * psi_n0
            # Run the multigroup Problem
            phi,psi_next = Source.multi_group(self,psi_last=psi_last,guess=phi_old)
            #if self.problem in ['ControlRod']:
            #    print('Time Step',t,'Flux',np.sum(phi),'\n===================================')
            # Update scalar/angular flux
            time_phi.append(phi)
            phi_old = phi.copy()
            psi_n0 = psi_n1.copy()
            psi_n1 = psi_next.copy()
            # Update source, change to BDF2 time steps
            self.lhs = full_lhs[t].copy(); self.td = 'BDF2'
        return phi,time_phi

    def surface_area(rho):
        return 4 * np.pi * rho**2

    def volume(plus,minus):
        return 4 * np.pi / 3 * (plus**3 - minus**3)

    def half_angle(psi_plus,total,delta,source):
        """ This is for finding the half angle (N = 1/2) at cell i """
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

