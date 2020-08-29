#!/usr/bin/env python

import numpy as numpy
import discrete1.setup as s
import discrete1.ae_prob as ae
from tensorflow import keras
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# autoencoder = keras.models.load_model('autoencoder/carbon15_source/model40-20-10_autoencoder.h5')
# encoder = keras.models.load_model('autoencoder/carbon15_source/model40-20-10_encoder.h5')
# decoder = keras.models.load_model('autoencoder/carbon15_source/model40-20-10_decoder.h5')
# coders = [autoencoder,encoder,decoder]

class source_auto:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L,R,I,track=False):
        self.G = G
        self.N = N
        self.mu = mu
        self.w = w
        self.total = total
        self.scatter = scatter
        self.chiNuFission = chiNuFission
        self.I = I
        self.inv_delta = float(I)/R
        self.track = track

    def scale_encode(self,matrix_full,atype):
        import numpy as np
        from discrete1.util import nnets
        if atype == 'fmult':
            model = self.fmult_encoder
            matrix,self.fmaxi,self.fmini = nnets.normalize(matrix_full,verbose=True)
        elif atype == 'smult':
            model = self.smult_encoder
            matrix,self.smaxi,self.smini = nnets.normalize(matrix_full,verbose=True)
        elif atype == 'phi':
            model = self.phi_encoder
            matrix,self.pmaxi,self.pmini = nnets.normalize(matrix_full,verbose=True)
        matrix[np.isnan(matrix)] = 0; 
        scale = np.sum(matrix,axis=1)
        matrix = model.predict(matrix)
        matrix = (scale/np.sum(matrix,axis=1))[:,None]*matrix
        matrix[np.isnan(matrix)] = 0;
        return matrix

    def phi_encoder(self,matrix_full,cell):
        import numpy as np
        from discrete1.util import nnets

        matrix,self.pmaxi[cell],self.pmini[cell] = nnets.normalize(matrix_full,verbose=True)
        matrix[np.isnan(matrix)] = 0; 
        scale = np.sum(matrix,axis=1)
        matrix = self.phi_encoder.predict(matrix)
        matrix = (scale/np.sum(matrix,axis=1))[:,None]*matrix
        matrix[np.isnan(matrix)] = 0;
        return matrix

    def one_group(self,smult,external,guess):
        import numpy as np
        from discrete1.util import nnets
        from discrete1.setup import func
        import ctypes

        cfunctions = ctypes.cdll.LoadLibrary('discrete1/data/clibrary.so')
        ae_sweep = clibrary.ae_sweep
        guess = guess.astype('float64')
        sources = (smult + external).astype('float64')
        total_xs = self.total.astype('float64')
        maxi = (self.pmaxi).astype('float64')
        mini = (self.pmini).astype('float64')

        phi = np.zeros((self.I,self.gprime),dtype='float64')
        for n in range(self.N):
            weight = (self.mu[n]*self.inv_delta).astype('float64')
            # psi_bottom = np.zeros((1,self.gprime)) # vacuum on LHS
            phi_ptr = ctypes.c_void_p(phi.ctypes.data)
            guess_ptr = ctypes.c_void_p(guess.ctypes.data)
            total_ptr = ctypes.c_void_p(total_xs.ctypes.data)
            source_ptr = ctypes.c_void_p(soures.ctypes.data)
            maxi_ptr = ctypes.c_void_p(maxi.ctypes.data)
            mini_ptr = ctypes.c_void_p(mini.ctypes.data)
            ae_sweep(phi_ptr,guess_ptr,total_ptr,source_ptr,maxi_ptr,mini.ptr,ctypes.c_double(weight))

            # Left to right
            # for ii in range(self.I):
            #     psi_top = source_auto.source_iteration(self,smult[ii],external[ii],psi_bottom,weight,guess[ii],ii)
            #     # print('Made it {}'.format(ii))
            #     phi[ii] = phi[ii] + (self.w[n] * func.diamond_diff(psi_top,psi_bottom))
            #     psi_bottom = psi_top.copy()
            # for ii in range(self.I-1,-1,-1):
            #     psi_top = psi_bottom.copy()
            #     psi_bottom = source_auto.source_iteration(self,smult[ii],external[ii],psi_top,weight,guess[ii],ii)
            #     # print('Made it {}'.format(ii))
            #     phi[ii] = phi[ii] +  (self.w[n] * func.diamond_diff(psi_top,psi_bottom))
        return phi

    def one_group(self,total,scatter,external,guess,tol=1e-08,MAX_ITS=100):
        """ Arguments:
            total: I x 1 vector of the total cross section for each spatial cell
            scatter: I x L+1 array for the scattering of the spatial cell by moment
            external: I array for the external sources
            guess: Initial guess of the scalar flux for a specific energy group (I x L+1)
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
        Returns:
            phi: a I array  """
        import numpy as np
        import ctypes
        clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cfunctions.so')
        sweep = clibrary.sweep
        converged = 0
        count = 1        
        phi_old = guess.copy()
        half_total = 0.5*total.copy()
        external = external.astype('float64')
        while not(converged):
            phi = np.zeros((self.I),dtype='float64')
            for n in range(self.N):
                weight = self.mu[n]*self.inv_delta
                top_mult = (weight-half_total).astype('float64')
                bottom_mult = (1/(weight+half_total)).astype('float64')
                temp_scat = (scatter * phi_old).astype('float64')
                # Set Pointers for C function
                phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)
                ext_ptr = ctypes.c_void_p(external.ctypes.data)
                top_ptr = ctypes.c_void_p(top_mult.ctypes.data)
                bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)
                sweep(phi_ptr,ts_ptr,ext_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]))
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi

    def source_iteration(self,mult,source,psi_bottom,weight,guess,cell):
        import numpy as np
        # old = np.random.rand(1,self.gprime) # initial guess
        old = guess[None,:].copy()
        converged = 0; count = 1
        while not (converged):
            alpha_top = source_auto.decodeTotalencode(self,old,cell)
            alpha_bottom = source_auto.decodeTotalencode(self,psi_bottom,cell)
            new = (mult + source + weight*psi_bottom + 0.5*alpha_bottom)*old/(weight*old-0.5*alpha_top)
            new[np.isnan(new)] = 0
            change = np.linalg.norm((new-old))
            # print('Change',change,'count',count)
            converged = (change < 1e-8) or (count >= 100)
            old = new.copy(); count += 1
        return new # of size (1 x G_hat)


    def decodeTotalencode(self,flux,ii):
        import numpy as np
        from discrete1.util import nnets

        scale = np.sum(flux)
        flux_full = self.phi_decoder.predict(flux)
        flux_full = (scale/np.sum(flux_full))*flux_full
        flux_full[np.isnan(flux_full)] = 0
        flux_full = nnets.unnormalize_single(flux_full,self.pmaxi[ii],self.pmini[ii])
        mult_full = self.total[ii]*flux_full
        mult_full,self.pmaxi[ii],self.pmini[ii] = nnets.normalize(mult_full,verbose=True)
        mult_full[np.isnan(mult_full)] = 0; 
        scale = np.sum(mult_full)
        mult = source_auto.phi_encoder(self,mult_full,ii)
        mult = (scale/np.sum(mult))*mult
        mult[np.isnan(mult)] = 0; 

        return mult

    def phi_decoder(self,flux):
        import numpy as np
        scale = np.sum(flux,axis=1)
        flux_full = self.phi_decoder.predict(flux)
        flux_full = (scale/np.sum(flux_full))[:,None]*flux_full
        flux_full[np.isnan(flux_full)] = 0
        flux_full = nnets.unnormalize(flux_full,self.pmaxi,self.pmini)
        return flux_full

    def transport(self,coder,problem='carbon',tol=1e-08,MAX_ITS=1000):
        import numpy as np
        from discrete1.util import nnets
        from discrete1.setup import ex_sources,func
        # Load Antoencoders, Encoders, Decoders
        # Phi
        phi_autoencoder,phi_encoder,phi_decoder = func.load_coder(coder)
        # self.phi_autoencoder = phi_autoencoder
        self.phi_decoder = phi_decoder; self.phi_encoder = phi_encoder
        # Scatter
        smult_autoencoder,smult_encoder,smult_decoder = func.load_coder(coder,ptype='smult')
        # self.smult_autoencoder = smult_autoencoder
        self.smult_decoder = smult_decoder; self.smult_encoder = smult_encoder
        # Fission
        fmult_autoencoder,fmult_encoder,fmult_decoder = func.load_coder(coder,ptype='fmult')
        # self.fmult_autoencoder = fmult_autoencoder
        self.fmult_decoder = fmult_decoder; self.fmult_encoder = fmult_encoder
        # Set Encoded Layer Dimension
        self.gprime = 20

        # Initialize phi
        phi_old_full = func.initial_flux(problem)
        # Initialize sources
        smult_full = np.einsum('ijk,ik->ij',self.scatter,phi_old_full)
        fmult_full = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old_full)
        print('Original Shapes',phi_old_full.shape,smult_full.shape,fmult_full.shape)

        # Encode Current Problems
        phi_old = source_auto.scale_encode(self,phi_old_full,atype='phi')
        smult = source_auto.scale_encode(self,smult_full,atype='smult')
        fmult = source_auto.scale_encode(self,smult_full,atype='fmult')
        mult = smult + fmult
        print('Encoded Shapes',phi_old.shape,smult.shape,fmult.shape)
        
        # How Am I Suppose to encode this?
        source = ex_sources.source1(self.I,self.G)
        source = self.phi_encoder.predict(source)

        converged = 0
        count = 1
        while not (converged):
            print('Source Iteration {}'.format(count))
            phi = source_auto.one_group(self,mult,source,phi_old)
            # Decode out phi
            phi_full = source_auto.phi_decoder(self,phi)
            # Calculate change
            change = np.linalg.norm((phi_full - phi_old_full)/phi_full/(self.I))
            # Check for convergence
            print('Change is',change,'\n===================================')    
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 
            # Continue with the next iterations
            phi_old_full = phi_full.copy()
            phi_old = phi.copy()
            # Calculate full mult matrices
            smult_full = np.einsum('ijk,ik->ij',self.scatter,phi_old_full)
            fmult_full = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old_full)
            # Encode down
            smult = source_auto.scale_encode(self,smult_full,atype='smult')
            fmult = source_auto.scale_encode(self,smult_full,atype='fmult')
            mult = smult + fmult

        return phi_full


problem = 'carbon'; enrich = 0.15
coders = 'autoencoder/carbon15_source/model40-20'
enrichment,splits = s.problem1.boundaries(enrich,problem=problem)

problem_ = source_auto(*s.problem1.variables(problem=problem))
# G,N,mu,w,total,scatter,fission,L,R,I = s.problem1.variables(problem=problem)
phi = problem_.transport(coders,problem=problem)