"""
Criticality Eigenvalue Problems
"""

from .KEproblems import Selection

import numpy as np
import ctypes

class Critical:
    # Keyword Arguments allowed currently
    __allowed = ("boundary") #,"track","enrich","split")

    def __init__(self,G,N,mu,w,total,scatter,fission,I,delta,**kwargs):
        """ Deals with Eigenvalue multigroup problems (reflected and vacuum boundaries)
        Attributes:
            G: Number of energy groups, int
            N: Number of discrete angles, int
            mu: Angles from Legendre Polynomials, numpy array of size (N,)
            w: Normalized weights for mu, numpy array of size (N,)
            total: total cross section, numpy array of size (I x G)
            scatter: scatter cross section, numpy array of size (I x G x G)
            fission: fission cross section, numpy array of size (I x G x G)
            I: number of spatial cells, int
            delta: width of each spatial cell, int
        kwargs:
            boundary: str (default reflected), determine RHS of problem
                options: 'vacuum', 'reflected'
            track: bool (default False), if track flux change with iteration
            enrich: float, used for labeling tracking # Combine with track?
            split: list of slices, used for tracking # Combine with track?
        """ 
        # Attributes
        self.G = G; self.N = N
        self.mu = mu; self.w = w
        self.total = total
        self.scatter = scatter
        self.fission = fission
        self.I = I
        self.delta = delta
        self.boundary = 'reflected'; 
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, available: boundary" 
            setattr(self, key, value)

        
    def one_group(self,total_,scatter_,source_,guess):
        """ Arguments:
            total_: I x 1 vector of the total cross section for each spatial cell
            scatter_: I x L+1 array for the scattering of the spatial cell by moment
            source_: I array for the external sources
            guess: Initial guess of the scalar flux for a specific energy group (I x L+1)
        Returns:
            phi: a I array  """
        clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cCritcal.so')
        sweep = clibrary.reflected
        if self.boundary == 'vacumm':
            print('Reflected')
            sweep = clibrary.vacuum
               
        phi_old = guess.copy()

        source_ = source_.astype('float64')
        ext_ptr = ctypes.c_void_p(source_.ctypes.data)

        tol=1e-12; MAX_ITS=100
        converged = 0; count = 1 
        while not(converged):
            phi = np.zeros((self.I),dtype='float64')
            for n in range(self.N):
                direction = ctypes.c_int(int(np.sign(self.mu[n])))
                weight = np.sign(self.mu[n]) * self.mu[n]*self.delta

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
    
    def update_q(self,phi,start,stop,g):
        return np.sum(self.scatter[:,g,start:stop]*phi[:,start:stop],axis=1)

    def multi_group(self,source,guess):
        phi_old = guess.copy()

        tol=1e-08; MAX_ITS=100
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)
            for g in range(self.G):
                q_tilde = source[:,g] + Critical.update_q(self,phi_old,g+1,self.G,g)
                if g != 0:
                    q_tilde += Critical.update_q(self,phi,0,g,g)

                phi[:,g] = Critical.one_group(self,self.total[:,g],self.scatter[:,g,g],q_tilde,phi_old[:,g])

            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi
            
    def power_iteration(self):

        phi_old = np.random.rand(self.I,self.G)
        phi_old /= np.linalg.norm(phi_old)

        sources = np.einsum('ijk,ik->ij',self.fission,phi_old) 

        tol=1e-12; MAX_ITS=100
        converged = 0; count = 1
        while not (converged):
            phi = Critical.multi_group(self,sources,phi_old)
            
            keff = np.linalg.norm(phi)
            phi /= keff
                        
            change = np.linalg.norm((phi-phi_old)/phi/(self.I))
            print('Change is',change,'Keff is',keff)

            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1

            phi_old = phi.copy()
            sources = np.einsum('ijk,ik->ij',self.fission,phi_old) 

        return phi,keff

    # def tracking_data(self,flux,sources=None):
    #     from discrete1.util import sn
    #     import numpy as np
    #     phi = flux.copy()
    #     # Scatter Tracking - separate phi and add label
    #     label_scatter = sn.cat(self.enrich,self.splits['scatter_djinn'])
    #     phi_scatter = sn.cat(phi,self.splits['scatter_djinn'])
    #     # Do not do this, results are inaccurate when normalized
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
    #     multiplier_fission = sn.cat(sources,self.splits['fission_djinn'])
    #     multiplier_full_fission = np.hstack((label_fission[:,None],multiplier_fission))
    #     fission_data = np.vstack((phi_full_fission[None,:,:],multiplier_full_fission[None,:,:]))
    #     return fission_data, scatter_data
