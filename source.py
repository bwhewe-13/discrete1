"""
Running Multigroup Source Problems
"""

from .setup_mg import Selection

import numpy as np
import ctypes

class Source:
    # Keyword Arguments allowed currently
    __allowed = ("boundary","time")

    def __init__(self,G,N,mu,w,total,scatter,fission,source,I,delta,**kwargs): 
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
        kwargs:
            boundary: str (default vacuum), determine RHS of problem
                options: 'vacuum', 'reflected'
            time: bool (default False), time dependency
            track: bool (default False), if track flux change with iteration
        """ 
        # Attributes
        self.G = G; self.N = N
        self.mu = mu; self.w = w
        self.total = total
        self.scatter = scatter
        self.fission = fission
        self.source = source
        self.I = I
        self.delta = delta
        # kwargs
        self.boundary = 'vacuum'; self.time = False
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, available: boundary, time" 
            setattr(self, key, value)

    @classmethod
    def stainless_infinite(cls,G,N):
        problem = cls(*Selection.select('stainless infinite',G,N))
        return problem

                
    def one_group(self,total_,scatter_,source_,guess):
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
            sweep = clibrary.reflected

        phi_old = guess.copy()
        
        source_ = source_.astype('float64')
        source_ptr = ctypes.c_void_p(source_.ctypes.data)

        tol = 1e-08; MAX_ITS = 100
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
                
                sweep(phi_ptr,ts_ptr,source_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]),direction)

            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()

        return phi

    def update_q(xs,phi,start,stop,g):
        return np.sum(xs[:,g,start:stop]*phi[:,start:stop],axis=1)

    def multi_group(self):
        """ Run multi group steady state problem
        Returns:
            phi: scalar flux, numpy array of size (I x G) """
        
        phi_old = np.random.rand(self.I,self.G)

        if self.time:
            print("Time")
        
        # source = np.einsum('ijk,ik->ij',self.fission,phi_old) + self.source
        # source += np.einsum('ijk,ik->ij',self.scatter,phi_old)

        tol = 1e-08; MAX_ITS = 100
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)  
            for g in range(self.G):
                q_tilde = self.source[:,g] + Source.update_q(self.scatter,phi_old,g+1,self.G,g) + Source.update_q(self.fission,phi_old,g+1,self.G,g)
                # q_tilde = Source.update_q(self.scatter,phi_old,g+1,self.G,g)
                if g != 0:
                    q_tilde = q_tilde + Source.update_q(self.scatter,phi,0,g,g) + Source.update_q(self.fission,phi,0,g,g)
                    # q_tilde = Source.update_q(self.scatter,phi,0,g,g)
                # phi[:,g] = Source.one_group(self,self.total[:,g],q_tilde,self.source[:,g],phi_old[:,g])
                phi[:,g] = Source.one_group(self,self.total[:,g],self.scatter[:,g,g]+self.fission[:,g,g],q_tilde,phi_old[:,g])
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            if np.isnan(change) or np.isinf(change):
                change = 0
            print('Change ',change,'\n===================================')
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 

            phi_old = phi.copy()
            # source = np.einsum('ijk,ik->ij',self.fission,phi) + self.source
            # source += np.einsum('ijk,ik->ij',self.scatter,phi)

        return phi

    def time_multi_group(self):
        return None


    def run(self):
        """ Will either call multi_group for steady state or time_multi_group
        for time dependency """
        if self.time:
            return Source.time_multi_group(self)
        return Source.multi_group(self)



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
 
