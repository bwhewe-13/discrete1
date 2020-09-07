#!/usr/bin/env python

import numpy as numpy
import discrete1.setup as s
import discrete1.ae_prob as ae
# from tensorflow import keras
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

    def multi_group(self,external,guess,count):
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
        from discrete1.util import nnets
        from discrete1.setup import func
        import ctypes
        import scipy.optimize as op

        forward_psi_true = np.zeros((self.N,self.I,self.G))
        forward_psi_approx = np.zeros((self.N,self.I,self.G))
        backward_psi_true = np.zeros((self.N,self.I,self.G))
        backward_psi_approx = np.zeros((self.N,self.I,self.G))

        phi = np.zeros((self.I,self.G),dtype='float64')
        phi_old = guess.copy()
        half_total = 0.5*self.total.copy()
        converged = 0; count = 1
        while not converged:
            smult = np.einsum('ijk,ik->ij',self.scatter,phi_old)
            phi = np.zeros((self.I,self.G),dtype='float64')
            for n in range(self.N):
                weight = (self.mu[n]*self.inv_delta).astype('float64')
                psi_bottom = np.zeros((1,self.G))
                top_mult = (weight-half_total)
                bottom_mult = (1/(weight+half_total))
                # Left to right
                for ii in range(self.I):
                    psi_top = source_auto.source_iteration(self,smult[ii],external[ii],psi_bottom,weight,guess[ii],ii)
                    psi_top[psi_top < 0] = 0
                    # forward_psi_approx[n,ii] = psi_top.flatten().copy()

                    # psi_top = (smult[ii] + external[ii] + psi_bottom * top_mult[ii]) *bottom_mult[ii]
                    # forward_psi_true[n,ii] = psi_top2.flatten().copy()

                    # Q = smult[ii]+external[ii] + weight*psi_bottom - 0.5*self.total[ii]*psi_bottom
                    # xs = self.total[ii].copy()
                    # moo = weight.copy()
                    # psi_top = op.fixed_point(source_auto.function,np.zeros((87)),args=(Q,xs,moo),xtol=1e-10,maxiter=100000)

                    phi[ii] = phi[ii] + (self.w[n] * func.diamond_diff(psi_top,psi_bottom))
                    psi_bottom = psi_top.copy()
                for ii in range(self.I-1,-1,-1):
                    psi_top = psi_bottom.copy()

                    psi_bottom = source_auto.source_iteration(self,smult[ii],external[ii],psi_top,weight,guess[ii],ii)
                    psi_bottom[psi_bottom < 0] = 0
                    # backward_psi_approx[n,ii] = psi_bottom.flatten().copy()

                    # psi_bottom = (smult[ii] + external[ii] + psi_bottom * top_mult[ii]) *bottom_mult[ii]
                    # backward_psi_true[n,ii] = psi_bottom2.flatten().copy()

                    # Q = smult[ii]+external[ii] + weight*psi_top - 0.5*self.total[ii]*psi_top
                    # xs = self.total[ii].copy()
                    # moo = weight.copy()
                    # psi_bottom = op.fixed_point(source_auto.function,np.zeros((87)),args=(Q,xs,moo),xtol=1e-10,maxiter=100000)

                    phi[ii] = phi[ii] +  (self.w[n] * func.diamond_diff(psi_top,psi_bottom))
            change = np.linalg.norm((phi-phi_old)/phi/(self.I))
            converged = (change < 1e-8) or (count >= 100)
            count += 1

            # Update to phi G
            phi_old = phi.copy()
        # np.save('angular_flux/forward_psi_true_{}'.format(str(count).zfill(3)),forward_psi_true)
        # np.save('angular_flux/forward_psi_approx_{}'.format(str(count).zfill(3)),forward_psi_approx)
        # np.save('angular_flux/backward_psi_true_{}'.format(str(count).zfill(3)),backward_psi_true)
        # np.save('angular_flux/backward_psi_approx_{}'.format(str(count).zfill(3)),backward_psi_approx)

        return phi

    def function(x,Q,total,weight):
        return (Q - 0.5*x*total)/weight


    def source_iteration(self,mult,source,psi_bottom,weight,guess,cell):
        import numpy as np

        old = guess[None,:].copy()

        converged = 0; count = 1
        alpha_bottom = self.total[cell] * psi_bottom
        
        new = np.zeros((1,self.G))
        while not (converged):
            alpha_top = self.total[cell] * old

            new = old*(mult+source+weight*psi_bottom-0.5*alpha_bottom)/(old*weight+0.5*alpha_top)

            new[new < -0.5] = 0
            new[np.isnan(new)] = 0
            change = np.argwhere(abs(old-new) < 1e-8)

            converged = (len(change) == 87) or (count >= 500)

            old = new.copy(); count += 1

        return new # of size (1 x G_hat)

            
    def transport(self,problem='carbon',tol=1e-12,MAX_ITS=100):
        """ Arguments:
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a I x G array    """        
        import numpy as np
        from discrete1.util import nnets
        from discrete1.setup import func

        phi_old = func.initial_flux(problem)
        
        # Initialize source
        sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old)

        converged = 0; count = 1
        while not (converged):
            print('Outer Transport Iteration {}\n==================================='.format(count))
            # Calculate Sigma_s * phi
            # smult = np.einsum('ijk,ik->ij',self.scatter,phi_old)
            
            # Calculate phi G'
            phi = source_auto.multi_group(self,sources,phi_old,count)
            # Convert to phi G, normalized

            keff = np.linalg.norm(phi)
            phi /= keff

            # Check for convergence with original phi sizes            
            change = np.linalg.norm((phi-phi_old)/phi/(self.I))
            print('Change is',change,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS)
            count += 1

            # Update to phi G
            phi_old = phi.copy()
            # Recalculate Sources
            sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old)

        return phi,keff


problem = 'carbon'; enrich = 0.15
# coders = 'autoencoder/carbon15_source/model40-20'
enrichment,splits = s.problem1.boundaries(enrich,problem=problem)

problem_ = source_auto(*s.problem1.variables(enrich,problem=problem))
# G,N,mu,w,total,scatter,fission,L,R,I = s.problem1.variables(problem=problem)
phi = problem_.transport(problem=problem)