
class eigen_eNDe:
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
        phi = np.zeros((self.I),dtype='float64')
        phi_old = guess.copy()
        half_total = 0.5*total.copy()
        external = external.astype('float64')
        while not(converged):
            # phi *= 0
            phi = np.zeros((self.I))
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
                # psi_bottom = 0 # vacuum on LHS
                # # Left to right
                # for ii in range(self.I):
                    # psi_top = (temp_scat[ii] + external[ii] + psi_bottom * (weight-half_total[ii]))/(weight+half_total[ii])
                    # phi[ii] = phi[ii] + (self.w[n] * func.diamond_diff(psi_top,psi_bottom))
                    # psi_bottom = psi_top
                # # Reflective right to left
                # for ii in range(self.I-1,-1,-1):
                    # psi_top = psi_bottom
                    # psi_bottom = (temp_scat[ii] + external[ii] + psi_top * (weight-half_total[ii]))/(weight+half_total[ii])
                    # phi[ii] = phi[ii] +  (self.w[n] * func.diamond_diff(psi_top,psi_bottom))
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi

    def multi_group(self,total,scatter,nuChiFission,guess,tol=1e-08,MAX_ITS=100):
        """ Arguments:
            total: I x G vector of the total cross section for each spatial cell and energy level
            scatter: I x G array for the scattering of the spatial cell by moment and energy
            nuChiFission: 
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
        Returns:
            phi: a I x G array  """
        import numpy as np
        from discrete1.util import nnets

        phi_old = guess.copy()
        # print(phi_old.shape)
        converged = 0
        count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)
            # smult = eigen_eNDe.update_q(self,scatter,phi_old)

            phi_old_full,self.pmaxi,self.pmini = nnets.normalize(phi_old,verbose=True)
            # print(phi_old.shape,phi_old_full.shape)
            phi_old_full = self.decoder.predict(phi_old_full)
            # print(phi_old.shape,phi_old_full.shape)
            phi_old_full = nnets.unnormalize(phi_old_full,self.pmaxi,self.pmini)
            
            smult = np.einsum('ijk,ik->ij',scatter,phi_old_full)

            smult,smaxi,smini = nnets.normalize(smult,verbose=True)
            smult[np.isnan(smult)] = 0
            smaxi[np.isnan(smaxi)] = 0; smini[np.isnan(smini)] = 0
            smult = self.encoder.predict(smult)
            smult = nnets.unnormalize(smult,smaxi,smini)

            for g in range(self.gprime):
                phi[:,g] = eigen_eNDe.one_group(self,total[:,g],smult[:,g],nuChiFission[:,g],tol=tol,MAX_ITS=MAX_ITS,guess=phi_old[:,g])
            # print(phi.shape,phi_old.shape,phi_old_full.shape)
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi
            
    def transport(self,coder,problem='carbon',tol=1e-12,MAX_ITS=100,LOUD=True,multAE=False):
        """ Arguments:
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a I x G array    """        
        import numpy as np
        from discrete1.util import nnets
        # from discrete1.util import sn
        autoencoder,encoder,decoder = func.load_coder(coder)
        self.gprime = func.find_gprime(coder)
        self.encoder = encoder; self.decoder = decoder

        phi_old = func.initial_flux(problem)
        phi_old_full = phi_old.copy()
        k_old = np.linalg.norm(phi_old)
        
        converged = 0
        count = 1

        # Calculate sources with original phi
        # sources = eigen_eNDe.update_q(self,self.chiNuFission,phi_old,small=False)
        # phi_old_small = eigen_eNDe.shrink(self,phi_old) 
        sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old)

        phi_old,self.pmaxi,self.pmini = nnets.normalize(phi_old,verbose=True)
        phi_old = self.encoder.predict(phi_old)
        phi_old = nnets.unnormalize(phi_old,self.pmaxi,self.pmini)
        
        sources,fmaxi,fmini = nnets.normalize(sources,verbose=True)
        sources[np.isnan(sources)] = 0;
        fmaxi[np.isnan(fmaxi)] = 0; fmini[np.isnan(fmini)] = 0
        sources = self.encoder.predict(sources)
        sources = nnets.unnormalize(sources,fmaxi,fmini)
        # print(phi_old.shape)
        while not (converged):
            print('Outer Transport Iteration {}\n==================================='.format(count))
            # Calculate phi G'
            phi = eigen_eNDe.multi_group(self,self.total,self.scatter,sources,phi_old,tol=1e-08,MAX_ITS=MAX_ITS)
            # Convert to phi G, normalized
            # phi,keff = eigen_eNDe.grow(self,phi,norm=True)
            # Convert to phi G
            phi,self.pmaxi,self.pmini = nnets.normalize(phi,verbose=True)
            phi = self.decoder.predict(phi)
            phi = nnets.unnormalize(phi,self.pmaxi,self.pmini)
            
            keff = np.linalg.norm(phi)
            phi /= keff

            # Check for convergence with original phi sizes            
            change = np.linalg.norm((phi-phi_old_full)/phi/(self.I))
            if LOUD:
                print('Change is',change,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS)
            count += 1

            # Update to phi G
            phi_old_full = phi.copy()

            sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old_full)

            phi_old_full,self.pmaxi,self.pmini = nnets.normalize(phi_old_full,verbose=True)
            phi_old = self.encoder.predict(phi_old_full)
            phi_old = nnets.unnormalize(phi_old,self.pmaxi,self.pmini)

            sources,fmaxi,fmini = nnets.normalize(sources,verbose=True)
            sources[np.isnan(sources)] = 0;
            fmaxi[np.isnan(fmaxi)] = 0; fmini[np.isnan(fmini)] = 0
            sources = self.encoder.predict(sources)
            sources = nnets.unnormalize(sources,fmaxi,fmini)

            # Update Sources and phi G'
            # sources = eigen_eNDe.update_q(self,self.chiNuFission,phi_old,small=False)
            # phi_old_small = eigen_eNDe.shrink(self,phi_old) 
        return phi,keff

class eigen_auto:
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
        phi = np.zeros((self.I),dtype='float64')
        phi_old = guess.copy()
        # phi_full = guess.copy()
        half_total = 0.5*total.copy()
        external = external.astype('float64')
        for n in range(self.N):
            weight = self.mu[n]*self.inv_delta
            top_mult = (weight-half_total).astype('float64')
            bottom_mult = (1/(weight+half_total)).astype('float64')
            # temp_scat = (scatter * phi_old).astype('float64')
            temp_scat = (scatter).astype('float64')
            # Set Pointers for C function
            phi_ptr = ctypes.c_void_p(phi.ctypes.data)
            ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)
            ext_ptr = ctypes.c_void_p(external.ctypes.data)
            top_ptr = ctypes.c_void_p(top_mult.ctypes.data)
            bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)
            sweep(phi_ptr,ts_ptr,ext_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]))

        return phi

    def multi_group(self,total,scatter,nuChiFission,guess,guess_full,tol=1e-08,MAX_ITS=10000):
        """ Arguments:
            total: I x G vector of the total cross section for each spatial cell and energy level
            scatter: I x G array for the scattering of the spatial cell by moment and energy
            nuChiFission: 
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
        Returns:
            phi: a I x G array  """
        import numpy as np
        from discrete1.util import nnets
        phi_old_full = guess_full.copy()
        phi_old = guess.copy()

        smult_full = np.einsum('ijk,ik->ij',scatter,phi_old_full)
        # Encode Scatter * Phi
        if self.multAE == 'scatter' or self.multAE == 'both':
            smult = eigen_auto.scale_autoencode(self,smult_full,atype='smult')
        else:
            smult = smult_full.copy()

        converged = 0
        count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)  
            for g in range(self.gprime):
                phi[:,g] = eigen_auto.one_group(self,total[:,g],smult[:,g],nuChiFission[:,g],phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
            # Decode out phi
            phi_full = eigen_auto.scale_autoencode(self,phi,atype='phi')

            change = np.linalg.norm((phi_full - phi_old_full)/phi_full/(self.I))
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 
            
            phi_old_full = phi_full.copy()
            phi_old = phi.copy()
            # Encode Scatter * Phi
            smult_full = np.einsum('ijk,ik->ij',scatter,phi_full)
            if self.multAE == 'scatter' or self.multAE == 'both':
                smult = eigen_auto.scale_autoencode(self,smult_full,atype='smult')
            else:
                smult = smult_full.copy()

        return phi_full
    
    def scale_autoencode(self,matrix_full,atype):
        import numpy as np
        from discrete1.util import nnets
        if atype == 'fmult':
            model = self.fmult_autoencoder
        elif atype == 'smult':
            model = self.smult_autoencoder
        elif atype == 'phi':
            model = self.phi_autoencoder
        matrix,maxi,mini = nnets.normalize(matrix_full,verbose=True)
        matrix[np.isnan(matrix)] = 0; maxi[np.isnan(maxi)] = 0; mini[np.isnan(mini)] = 0
        scale = np.sum(matrix,axis=1)
        matrix = model.predict(matrix)
        matrix = (scale/np.sum(matrix,axis=1))[:,None]*matrix
        matrix[np.isnan(matrix)] = 0;
        matrix = nnets.unnormalize(matrix,maxi,mini)
        return matrix

    def transport(self,coder,problem='carbon',tol=1e-12,MAX_ITS=100,LOUD=True,multAE=False):
        """ Arguments:
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a I x G array    """        
        import numpy as np
        from discrete1.util import nnets
        from discrete1.theProcess import func
        # from discrete1.util import sn
        phi_autoencoder,phi_encoder,phi_decoder = func.load_coder(coder)
        self.phi_autoencoder = phi_autoencoder
        
        self.multAE = multAE
        self.gprime = 87

        if self.multAE:
            smult_autoencoder,smult_encoder,smult_decoder = func.load_coder(coder,ptype='smult')
            self.smult_autoencoder = smult_autoencoder
            fmult_autoencoder,fmult_encoder,fmult_decoder = func.load_coder(coder,ptype='fmult')
            self.fmult_autoencoder = fmult_autoencoder

        phi_old_full = func.initial_flux(problem)
        keff = np.linalg.norm(phi_old_full)

        sources_full = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old_full)
        # Encode-Decode Phi
        phi_old = eigen_auto.scale_autoencode(self,phi_old_full,atype='phi')
        # Unnormalized Method
        # phi_old = self.phi_autoencoder.predict(phi_old_full)
        # phi_old = (np.sum(phi_old_full,axis=1)/np.sum(phi_old,axis=1))[:,None]*phi_old
        if self.multAE == 'fission' or self.multAE == 'both':
            sources = eigen_auto.scale_autoencode(self,sources_full,atype='fmult')
        else:
            sources = sources_full.copy()

        converged = 0
        count = 1
        while not (converged):
            print('Outer Transport Iteration {}'.format(count))
            phi = eigen_auto.multi_group(self,self.total,self.scatter,sources,phi_old,phi_old_full,tol=1e-08,MAX_ITS=MAX_ITS)
            # Expand to correct size
            phi_full = eigen_auto.scale_autoencode(self,phi,atype='phi')
            
            keff = np.linalg.norm(phi_full)
            phi_full /= keff

            change = np.linalg.norm((phi_full-phi_old_full)/phi_full/(self.I))
            # change = np.linalg.norm((phi-phi_old)/phi/(self.I))
            if LOUD:
                print('Change is',change,'Keff is',keff)
                print('===================================')
            converged = (change < tol) or (count >= MAX_ITS) #or (kchange < tol)
            count += 1

            phi_old_full = phi_full.copy()
            phi_old = phi.copy()

            sources_full = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old_full)
            if self.multAE == 'fission' or self.multAE == 'both':
                sources = eigen_auto.scale_autoencode(self,sources_full,atype='fmult')
            else:
                sources = sources_full.copy()

        return phi_full,keff

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
        phi = np.zeros((self.I),dtype='float64')
        phi_old = guess.copy()
        # phi_full = guess.copy()
        half_total = 0.5*total.copy()
        external = external.astype('float64')
        phi = np.zeros((self.I))
        for n in range(self.N):
            weight = self.mu[n]*self.inv_delta
            top_mult = (weight-half_total).astype('float64')
            bottom_mult = (1/(weight+half_total)).astype('float64')
            # temp_scat = (scatter * phi_old).astype('float64')
            temp_scat = (scatter).astype('float64')
            # Set Pointers for C function
            phi_ptr = ctypes.c_void_p(phi.ctypes.data)
            ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)
            ext_ptr = ctypes.c_void_p(external.ctypes.data)
            top_ptr = ctypes.c_void_p(top_mult.ctypes.data)
            bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)
            sweep(phi_ptr,ts_ptr,ext_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]))
        return phi



    def scale_autoencode(self,matrix_full,atype):
        import numpy as np
        from discrete1.util import nnets
        if atype == 'fmult':
            model = self.fmult_autoencoder
        elif atype == 'smult':
            model = self.smult_autoencoder
        elif atype == 'phi':
            model = self.phi_autoencoder
        matrix,maxi,mini = nnets.normalize(matrix_full,verbose=True)
        matrix[np.isnan(matrix)] = 0; maxi[np.isnan(maxi)] = 0; mini[np.isnan(mini)] = 0
        scale = np.sum(matrix,axis=1)
        matrix = model.predict(matrix)
        matrix = (scale/np.sum(matrix,axis=1))[:,None]*matrix
        matrix[np.isnan(matrix)] = 0;
        matrix = nnets.unnormalize(matrix,maxi,mini)
        return matrix

    def creating_mult(self,fmult_full,smult_full):
        if self.multAE == 'fission':
            fmult = source_auto.scale_autoencode(self,fmult_full,atype='fmult')
            smult = smult_full.copy()
        elif self.multAE == 'scatter':
            fmult = fmult_full.copy()
            smult = source_auto.scale_autoencode(self,smult_full,atype='smult')
        elif self.multAE == 'both':
            fmult = source_auto.scale_autoencode(self,fmult_full,atype='fmult')
            smult = source_auto.scale_autoencode(self,smult_full,atype='smult')
        else:
            fmult = fmult_full.copy(); smult = smult_full.copy()
        # Combine Source terms and return
        return smult + fmult

    def transport(self,coder,problem='carbon',tol=1e-08,MAX_ITS=1000,multAE=False):
        """ Arguments:
            total: I x G vector of the total cross section for each spatial cell and energy level
            scatter: I x G array for the scattering of the spatial cell by moment and energy
            nuChiFission: 
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
        Returns:
            phi: a I x G array  """
        import numpy as np
        from discrete1.util import nnets
        from discrete1.theProcess import ex_sources,func
        phi_autoencoder,phi_encoder,phi_decoder = func.load_coder(coder)
        self.phi_autoencoder = phi_autoencoder

        self.multAE = multAE
        self.gprime = 87

        if self.multAE:
            smult_autoencoder,smult_encoder,smult_decoder = func.load_coder(coder,ptype='smult')
            self.smult_autoencoder = smult_autoencoder
            fmult_autoencoder,fmult_encoder,fmult_decoder = func.load_coder(coder,ptype='fmult')
            self.fmult_autoencoder = fmult_autoencoder
        # Initialize phi
        phi_old_full = func.initial_flux(problem)
        # Initialize sources
        smult_full = np.einsum('ijk,ik->ij',self.scatter,phi_old_full)
        fmult_full = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old_full)
        # Encode-Decode Phi
        phi_old = source_auto.scale_autoencode(self,phi_old_full,atype='phi')
        # phi_old = self.autoencoder.predict(phi_old_full)
        # phi_old = (np.sum(phi_old_full,axis=1)/np.sum(phi_old,axis=1))[:,None]*phi_old
        mult = source_auto.creating_mult(self,fmult_full,smult_full)
        source = ex_sources.source1(self.I,self.G)

        converged = 0
        count = 1
        while not (converged):
            print('Source Iteration {}'.format(count))
            phi = np.zeros(phi_old.shape)  
            for g in range(self.gprime):
                phi[:,g] = source_auto.one_group(self,self.total[:,g],mult[:,g],source,phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
            # Decode out phi
            phi_full = source_auto.scale_autoencode(self,phi,atype='phi')
            # Check for convergence
            change = np.linalg.norm((phi_full - phi_old_full)/phi_full/(self.I))
            
            print('Change is',change,'\n===================================')    
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 

            phi_old_full = phi_full.copy()
            phi_old = phi.copy()
            
            smult_full = np.einsum('ijk,ik->ij',self.scatter,phi_old_full)
            fmult_full = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old_full)
            mult = source_auto.creating_mult(self,fmult_full,smult_full)
        return phi_full
            
