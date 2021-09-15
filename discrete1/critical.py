""" Criticality Eigenvalue Problems """

from .keigenvalue import Problem1, Problem2, Problem3, Problem4
from .reduction import DJ,AE,DJAE

import numpy as np
import ctypes
import warnings
import os
warnings.filterwarnings("ignore", category=RuntimeWarning) 

DATA_PATH = 'discrete1/data/'

class Critical:
    # Keyword Arguments allowed currently
    __allowed = ("boundary","track","geometry","reduced")

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
            geometry: str (default slab), determines coordinates of sweep ('slab' or 'sphere')
            reduce: list of cross sections instead of each spatial cell
        """ 
        # Attributes
        self.G = G; self.N = N
        self.mu = mu; self.w = w
        self.total = total
        self.scatter = scatter
        self.fission = fission
        self.I = I
        self.delta = 1/delta
        self.boundary = 'reflected'; self.track = False
        self.atype = ''; self.saving = '0'
        self.geometry = 'slab'
        self.reduced = None
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, available: boundary, track, geometry, reduced" 
            setattr(self, key, value)
        Critical.compile(I,N)


    @classmethod
    def run(cls,refl,enrich,orient='orig',**kwargs):
        # This is for running the preset eigenvalue problems
        if refl in ['hdpe','ss440']:
            attributes = Problem1.steady(refl,enrich,orient)
        elif refl in ['pu']:
            attributes = Problem2.steady('hdpe',enrich,orient)
        elif refl in ['c']:
            groups = kwargs['G'] if 'G' in kwargs else 87
            attributes = Problem3.steady('hdpe',enrich,orient,groups)
        elif refl in ['diff']:
            attributes = Problem4.steady('ss440',enrich,orient,kwargs['sn'])

        boundary = kwargs['boundary'] if 'boundary' in kwargs else 'reflected'
        problem = cls(*attributes,boundary=boundary,geometry='sphere')
        problem.saving = '0'
        problem.atype = ''
        return problem.transport()

    @classmethod
    def run_djinn(cls,refl,enrich,models,atype,orient='orig',**kwargs):
        if refl in ['hdpe','ss440']:
            attributes = Problem1.steady(refl,enrich,orient)
        elif refl in ['pu']:
            attributes = Problem2.steady('hdpe',enrich,orient)
        initial = '{}/initial_{}.npy'.format(DATA_PATH,refl)

        problem = cls(*attributes)
        problem.saving = '1' # To know when to call DJINN
        problem.atype = atype
        problem.initial = initial

        model = DJ(models,atype,**kwargs)
        model.load_model()
        model.load_problem(refl,enrich,orient)

        return problem.transport(models=model)

    @classmethod
    def run_auto(cls,refl,enrich,models,atype,orient='orig',transform='cuberoot',**kwargs):
        if refl in ['hdpe','ss440']:
            attributes = Problem1.steady(refl,enrich,orient)
        elif refl in ['pu']:
            attributes = Problem2.steady('hdpe',enrich,orient)
        initial = '{}/initial_{}.npy'.format(DATA_PATH,refl)

        problem = cls(*attributes)
        problem.saving = '2'
        problem.atype = atype
        problem.initial = initial

        model = AE(models,atype,transform,**kwargs)
        model.load_model()
        model.load_problem(refl,enrich,orient)

        return problem.transport(models=model)

    @classmethod
    def run_djae(cls,refl,enrich,dj_models,ae_models,atype,orient='orig',transform='cuberoot',**kwargs):
        if refl in ['hdpe','ss440']:
            attributes = Problem1.steady(refl,enrich,orient)
        elif refl in ['pu']:
            attributes = Problem2.steady('hdpe',enrich,orient)
        initial = '{}/initial_{}.npy'.format(DATA_PATH,refl)
        # initial = 'mydata/djinn_pluto_perturb/true_phi_{}_15.npy'.format(orient)
        problem = cls(*attributes)
        problem.saving = '3' # To know when to call DJINN
        problem.atype = atype
        problem.initial = initial
        model = DJAE(dj_models,ae_models,atype,transform,**kwargs)
        model.load_model()
        model.load_problem(refl,enrich,orient)
        return problem.transport(models=model)

        
    @classmethod
    def run_reduce(cls,refl,enrich,orient='orig',**kwargs):
        # This is only for slab problems 
        attributes = Problem2.steady('hdpe',enrich,orient)
        _,splits = Problem2.labeling('hdpe',enrich,orient)
        splits = np.sort(np.array([item for sublist in list(splits.values()) for item in sublist]))

        boundary = kwargs['boundary'] if 'boundary' in kwargs else 'reflected'
        problem = cls(*attributes,boundary=boundary,reduced=splits)
        
        global_indices = [len(range(*ii.indices(problem.I))) for ii in splits]
        cum_indices = np.insert(np.cumsum(global_indices,dtype=int),0,0)[:-1]

        problem.scatter = problem.scatter[cum_indices].copy()
        problem.fission = problem.fission[cum_indices].copy()
        problem.total = problem.total[cum_indices].copy()

        problem.saving = '0'
        problem.atype = ''

        Critical.reduce_compile(problem.I,global_indices)
        return problem.transport()


    def slab(self,total_,scatter_,source_,guess,tol=1e-08,MAX_ITS=100):
        """ Arguments:
            total_: I x 1 vector of the total cross section for each spatial cell
            scatter_: I x L+1 array for the scattering of the spatial cell by moment
            source_: I array for the external sources
            guess: Initial guess of the scalar flux for a specific energy group (I x L+1)
        Returns:
            phi: a I array  """
        clibrary = ctypes.cdll.LoadLibrary('./{}/cCritical.so'.format(DATA_PATH))
        sweep = clibrary.reflected if self.boundary == 'reflected' else clibrary.vacuum

        phi_old = guess.copy()
        source_ = source_.astype('float64')
        ext_ptr = ctypes.c_void_p(source_.ctypes.data)

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

                if self.atype in ['scatter','both'] or self.reduced is not None:
                    temp_scat = (scatter_).astype('float64')
                else:
                    temp_scat = (scatter_ * phi_old).astype('float64')

                ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)
                phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                
                if self.reduced is not None:
                    phi_old = phi_old.astype('float64')
                    gu_ptr = ctypes.c_void_p(phi_old.ctypes.data)                    
                    clibrary.reflected_reduced(phi_ptr,gu_ptr,ts_ptr,ext_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]))
                else:
                    sweep(phi_ptr,ts_ptr,ext_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]),direction)

            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()

        return phi

    def sphere(self,total_,scatter_,source_,guess,tol=1e-08,MAX_ITS=100):
        """ Arguments:
            total_: I x 1 vector of the total cross section for each spatial cell
            scatter_: I x L+1 array for the scattering of the spatial cell by moment
            source_: I array for the external sources
            guess: Initial guess of the scalar flux for a specific energy group (I x L+1)
        Returns:
            phi: a I array  """
        clibrary = ctypes.cdll.LoadLibrary('./{}/cCriticalSP.so'.format(DATA_PATH))
        sweep = clibrary.vacuum

        edges = np.cumsum(np.insert(np.ones((self.I))*1/self.delta,0,0))
        SA_plus = Critical.surface_area(edges[1:]).astype('float64')       # Positive surface area
        SAp_ptr = ctypes.c_void_p(SA_plus.ctypes.data)

        SA_minus = Critical.surface_area(edges[:self.I]).astype('float64') # Negative surface area
        SAm_ptr = ctypes.c_void_p(SA_minus.ctypes.data)

        V = Critical.volume(edges[1:],edges[:self.I])                      # Volume and total
        v_total = (V * total_).astype('float64')
        v_ptr = ctypes.c_void_p(v_total.ctypes.data)

        phi_old = guess.copy()

        mu = self.mu.astype('float64')                                     # Angle and weight
        mu_ptr = ctypes.c_void_p(mu.ctypes.data)
        w = self.w.astype('float64')
        w_ptr = ctypes.c_void_p(w.ctypes.data)

        converged = 0; count = 1;

        # psi_nhalf = (Critical.half_angle(0,total_,1/self.delta, source_ + scatter_ * phi_old)).astype('float64')
        while not (converged):
            phi = np.zeros((self.I),dtype='float64')
            if self.atype in ['scatter','both']:
                temp = scatter_
            else:
                temp = scatter_ * phi_old
            psi_nhalf = (Critical.half_angle(0,total_,1/self.delta, source_ + temp)).astype('float64')

            Q = (V * (source_ + temp)).astype('float64')
            q_ptr = ctypes.c_void_p(Q.ctypes.data)

            psi_ptr = ctypes.c_void_p(psi_nhalf.ctypes.data)
            phi_ptr = ctypes.c_void_p(phi.ctypes.data)

            sweep(phi_ptr,psi_ptr,q_ptr,v_ptr,SAp_ptr,SAm_ptr,mu_ptr,w_ptr)
            
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) #or (change == np.nan)
            count += 1
            phi_old = phi.copy()
            # print(change)

        return phi

    
    def update_q(self,phi,start,stop,g):
        if self.reduced is not None:
            return np.sum(Critical.repopulate(self.reduced,self.scatter[:,g,start:stop],phi[:,start:stop]),axis=1)
        return np.sum(self.scatter[:,g,start:stop]*phi[:,start:stop],axis=1)

    def multi_group(self,source,guess,tol=1e-12,MAX_ITS=100):
        phi_old = guess.copy()

        geo = getattr(Critical,self.geometry)  # Get the specific sweep

        converged = 0; count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)
            if self.atype in ['scatter','both']:
                phi_old[np.isnan(phi_old)] = 0
                smult = Critical.sorting_scatter(self,phi_old,self.models)
                for g in range(self.G):
                    phi[:,g] = geo(self,self.total[:,g],smult[:,g],source[:,g],phi_old[:,g])
            else:
                
                for g in range(self.G):
                    q_tilde = source[:,g] + Critical.update_q(self,phi_old,g+1,self.G,g)
                    if g != 0:
                        q_tilde += Critical.update_q(self,phi,0,g,g)
                    phi[:,g] = geo(self,self.total[:,g],self.scatter[:,g,g],q_tilde,phi_old[:,g])
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi
            
    def transport(self,models=None,tol=1e-12,MAX_ITS=100):

        self.models = models
        # if self.saving != '0': # Running from random
        if self.saving not in ['0','2']:
            phi_old = np.load(self.initial)
        else:
            phi_old = np.random.rand(self.I,self.G)
            phi_old /= np.linalg.norm(phi_old)
            
        converged = 0; count = 1
        while not (converged):
            sources = Critical.sorting_fission(self,phi_old,self.models)
            phi_old = Critical.sorting_phi(self,phi_old,self.models)

            print('Outer Transport Iteration {}\n==================================='.format(count))
            phi = Critical.multi_group(self,sources,phi_old)
            
            keff = np.linalg.norm(phi)
            phi /= keff

            change = np.linalg.norm((phi-phi_old)/phi/(self.I))
            print('Change is',change,'Keff is',keff)

            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1

            phi_old = phi.copy()

        return phi,keff

    def sorting_fission(self,phi,models):
        if self.reduced is not None:
            return Critical.repopulate(self.reduced,self.fission,phi)
        if self.saving == '0' or self.atype not in ['both','fission']: # No Reduction
            return np.einsum('ijk,ik->ij',self.fission,phi)
        elif self.saving in ['1','3']:                                 # DJINN / DJINN + Autoencoder
            # return np.einsum('ijk,ik->ij',self.fission,phi)
            return models.predict_fission(phi)
        elif self.saving in ['2']:                                     # Autoencoder Squeeze
            return models.squeeze(np.einsum('ijk,ik->ij',self.fission,phi))

    def sorting_scatter(self,phi,models):
        if self.saving in ['1','3']:                      # DJINN / DJINN + Autoencoder
            return models.predict_scatter(phi)
        elif self.saving in ['2']:                        # Autoencoder Squeeze
            return models.squeeze(np.einsum('ijk,ik->ij',self.scatter,phi))

    def sorting_phi(self,phi,models):
        if self.saving in ['2'] and self.atype == 'phi':
            return models.squeeze(phi)
        else:
            return phi

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

    def compile(I,N):
        # Compile Slab
        command = 'gcc -fPIC -shared -o {}cCritical.so {}cCritical.c -DLENGTH={}'.format(DATA_PATH,DATA_PATH,I)
        os.system(command)
        # Compile Sphere
        command = 'gcc -fPIC -shared -o {}cCriticalSP.so {}cCriticalSP.c -DLENGTH={} -DN={}'.format(DATA_PATH,DATA_PATH,I,N)
        os.system(command)

    def reduce_compile(I,materials):
        # Compile Slab
        extraneous = ' -DHDPE={} -DPU239={} -DPU240={}'.format(materials[0],materials[1],materials[2])
        command = 'gcc -fPIC -shared -o {}cCritical.so {}cCritical.c -DLENGTH={}'.format(DATA_PATH,DATA_PATH,I)
        command += extraneous
        os.system(command)

    def repopulate(index,xs,phi):
        # Multiplying full phi by reduced cross section
        if len(xs.shape) == 2:  # Total cross section
            return np.concatenate([xs[ii] * phi[index[ii]] for ii in range(len(index))])
        return np.concatenate([phi[index[ii]] @ xs[ii].T for ii in range(len(index))])

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
