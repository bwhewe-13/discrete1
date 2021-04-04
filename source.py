"""
Running Multigroup Source Problems
"""

from .fixed import FixedSource

import numpy as np
import ctypes

class Source:
    # Keyword Arguments allowed currently
    __allowed = ("T","dt","v","boundary","geometry") # ,"hybrid","enrich","track")

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
        self.geometry = 'slab'
        #self.enrich = None; self.time = False
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, available: boundary, time, geometry" 
            setattr(self, key, value)

    @classmethod
    def run(cls,ptype,G,N,**kwargs):
        # Default is vacuum
        boundary = 'vacuum'
        if 'boundary' in kwargs:
            boundary = kwargs['boundary']
        if 'geometry' in kwargs:
            geometry = kwargs['geometry']
        # Currently cannot use reflected boundary with TD problems
        if 'T' in kwargs and boundary == 'reflected':
            print('Using Vacuum Boundaries, cannot use Time Dependency with Reflected Conditions')
            boundary = 'vacuum'

        attributes,keywords = FixedSource.initialize(ptype,G,N,**kwargs)

        problem = cls(*attributes,**keywords)

        problem.boundary = boundary
        problem.problem = ptype
        problem.geometry = geometry
        
        if 'T' in kwargs:
            return problem.time_steps()

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
            angular = np.zeros((self.I),dtype='float64')
            an_ptr = ctypes.c_void_p(angular.ctypes.data)

            phi = np.zeros((self.I),dtype='float64')
            # psi_nhalf = (Source.half_angle(0,total_,1/self.delta, source_ + scatter_ * phi_old)).astype('float64')
            for n in range(self.N):
                if n == 0:
                    alpha_minus = ctypes.c_double(0.)
                    psi_nhalf = (Source.half_angle(0,total_,1/self.delta, source_ + scatter_ * phi_old)).astype('float64')
                if n == self.N - 1:
                    alpha_plus = ctypes.c_double(0.)
                else:
                    alpha_plus = ctypes.c_double(alpha_minus - self.mu[n] * self.w[n])

                # psi_ihalf = ctypes.c_double(min(0.,psi_centers[N-n-1],key=abs))
                if self.mu[n] > 0:
                    psi_ihalf = ctypes.c_double(psi_centers[self.N-n-1])
                elif self.mu[n] < 0:
                    psi_ihalf = ctypes.c_double(boundary)

                Q = (V * (source_ + scatter_ * phi_old)).astype('float64')
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
        v_total = (V * total_ + speed).astype('float64')
        v_ptr = ctypes.c_void_p(v_total.ctypes.data)

        phi_old = guess.copy()
        psi_next = np.zeros(psi_last.shape,dtype='float64')
        psi_centers = np.zeros((self.N),dtype='float64')

        tol=1e-12; MAX_ITS=100
        converged = 0; count = 1; 
        while not (converged):
            angular = np.zeros((self.I),dtype='float64')
            an_ptr = ctypes.c_void_p(angular.ctypes.data)

            phi = np.zeros((self.I),dtype='float64')
            # psi_nhalf = (Source.half_angle(0,total_,1/self.delta, source_ + scatter_ * phi_old)).astype('float64')
            for n in range(self.N):
                if n == 0:
                    alpha_minus = ctypes.c_double(0.)
                    psi_nhalf = (Source.half_angle(0,total_,1/self.delta, source_ + scatter_ * phi_old)).astype('float64')
                if n == self.N - 1:
                    alpha_plus = ctypes.c_double(0.)
                else:
                    alpha_plus = ctypes.c_double(alpha_minus - self.mu[n] * self.w[n])

                # psi_ihalf = ctypes.c_double(min(0.,psi_centers[N-n-1],key=abs))
                if self.mu[n] > 0:
                    psi_ihalf = ctypes.c_double(psi_centers[self.N-n-1])
                elif self.mu[n] < 0:
                    psi_ihalf = ctypes.c_double(boundary)

                Q = (V * (source_ + scatter_ * phi_old) + psi_last[:,n] * speed).astype('float64') #+ psi_last[:,n] * speed
                q_ptr = ctypes.c_void_p(Q.ctypes.data)

                psi_ptr = ctypes.c_void_p(psi_nhalf.ctypes.data)
                phi_ptr = ctypes.c_void_p(phi.ctypes.data)

                clib.sweep(an_ptr,phi_ptr,psi_ptr,q_ptr,v_ptr,SAp_ptr,SAm_ptr,ctypes.c_double(self.w[n]),ctypes.c_double(self.mu[n]),alpha_plus,alpha_minus,psi_ihalf)
                # Update angular center corrections
                psi_centers[n] = angular[0]
                psi_next[:,n] = angular.copy()
                # Update angular difference coefficients
                alpha_minus = alpha_plus
                
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

                top_mult = (weight - 0.5 * total_ - 0.5 * speed).astype('float64')
                top_ptr = ctypes.c_void_p(top_mult.ctypes.data)

                bottom_mult = (1/(weight + 0.5 * total_ + 0.5 * speed)).astype('float64')
                bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)

                temp_scat = (scatter_ * phi_old).astype('float64')
                ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)
                
                phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                
                sweep(phi_ptr,psi_ptr,ts_ptr,rhs_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]),lhs,direction)

                psi_next[:,n] = psi_angle.copy()

            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            if np.isnan(change) or np.isinf(change):
                change = 0.
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()

        return phi,psi_next

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
                q_tilde = self.source[:,g] + Source.update_q(self.scatter,phi_old,g+1,self.G,g) + Source.update_q(self.fission,phi_old,g+1,self.G,g)
                if g != 0:
                    q_tilde += Source.update_q(self.scatter,phi_old,0,g,g) + Source.update_q(self.fission,phi,0,g,g)
                if self.T:
                    phi[:,g],psi_next[:,:,g] = geo(self,self.total[:,g],self.scatter[:,g,g]+self.fission[:,g,g],q_tilde,self.lhs[g],phi_old[:,g],psi_last[:,:,g],self.speed[g])
                else:
                    phi[:,g] = geo(self,self.total[:,g],self.scatter[:,g,g]+self.fission[:,g,g],q_tilde,self.lhs[g],phi_old[:,g])

            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            if np.isnan(change) or np.isinf(change):
                change = 0.5
            if not self.T:
                print('Count',count,'Change',change,'\n===================================')
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 

            phi_old = phi.copy()

        if self.T:
            return phi,psi_next
        return phi

    def time_steps(self):
        phi_old = np.zeros((self.I,self.G))
        psi_last = np.zeros((self.I,self.N,self.G))

        self.speed = 1/(self.v*self.dt); time_phi = []

        steps = int(self.T/self.dt)
        for t in range(int(self.T/self.dt)):
            # Solve at initial time step
            phi,psi_next = Source.multi_group(self,psi_last=psi_last,guess=phi_old)

            print('Time Step',t,'Flux',np.sum(phi),'\n===================================')
            # Update angular flux
            psi_last = psi_next.copy()
            time_phi.append(phi)
            phi_old = phi.copy()

            if self.problem in ['Stainless','UraniumStainless','StainlessUranium']: # and t > 2:
                if t < int(0.2*steps):
                    self.lhs *= 1
                elif t % int(0.1*steps) == 0:
                    self.lhs *= 0.5

        return phi,time_phi

    def surface_area(rho):
        return 4 * np.pi * rho**2

    def volume(plus,minus):
        return 4 * np.pi / 3 * (plus**3 - minus**3)

    def half_angle(psi_plus,total,delta,source):
        """ This is for finding the half angle (N = 1/2) at cell i """
        return (2 * psi_plus + delta * source ) / (2 + total * delta)

    # def tracking_data(self,flux,sources=None):
    #     from discrete1.util import sn
    #     import numpy as np
    #     # Normalize phi
    #     phi = flux.copy()
    #     # phi /= np.linalg.norm(phi)
    #     # Scatter Tracking - separate phi and add label
    #     label_scatter = sn.cat(self.enrich,self.splits['scatter_djinn'])
    #     phi_scatter = sn.cat(phi,self.splits['scatter_djinn'])
    #     # phi_scatter /= np.linalg.norm(phi_scatter)
    #     phi_full_scatter = np.hstack((label_scatter[:,None],phi_scatter))
    #     # Separate scatter multiplier and add label
    #     multiplier_scatter = np.einsum('ijk,ik->ij',sn.cat(self.scatter,self.splits['scatter_djinn']),phi_scatter)
    #     multiplier_full_scatter = np.hstack((label_scatter[:,None],multiplier_scatter))
    #     scatter_data = np.vstack((phi_full_scatter[None,:,:],multiplier_full_scatter[None,:,:]))
    #     # Fission Tracking - Separate phi and add label
    #     label_fission = sn.cat(self.enrich,self.splits['fission_djinn'])
    #     phi_fission = sn.cat(phi,self.splits['fission_djinn'])
    #     phi_full_fission = np.hstack((label_fission[:,None],phi_fission))
    #     # Separate fission multiplier and add label
    #     # multiplier_fission = sn.cat(sources,self.splits['fission_djinn'])
    #     multiplier_fission = np.einsum('ijk,ik->ij',sn.cat(self.chiNuFission,self.splits['fission_djinn']),phi_fission)
    #     multiplier_full_fission = np.hstack((label_fission[:,None],multiplier_fission))
    #     fission_data = np.vstack((phi_full_fission[None,:,:],multiplier_full_fission[None,:,:]))
    #     return fission_data, scatter_data
