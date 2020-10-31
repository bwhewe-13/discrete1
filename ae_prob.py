
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
        
    def multi_group(self,smult,external,guess):
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
        # import scipy.optimize as op

        # clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/clibrary_ae.so')
        # ae_sweep = clibrary.sweep_ae
        # print('Here in function')
        # guess = guess.astype('float64')
        
        # sources = (smult + external).astype('float64')
        # total_xs = self.total.astype('float64')
        # maxi = (self.pmaxi).astype('float64')
        # mini = (self.pmini).astype('float64')
        # half_total = 0.5*self.total.copy()
        # left_psis = np.load('discrete1/data/left_psis.npy')
        # right_psis = np.load('discrete1/data/right_psis.npy')
        phi_old = guess.copy()
        converged = 0; count = 1
        # print('mult sum',np.sum(mult))
        while not converged:
            phi = np.zeros((self.I,self.gprime),dtype='float64')
            for n in range(self.N):
                weight = (self.mu[n]*self.inv_delta).astype('float64')
                psi_bottom = np.zeros((1,self.gprime))
                # phi = np.zeros((self.I,self.gprime),dtype='float64')
                # psi_bottom = np.zeros((1,self.gprime)) # vacuum on LHS
                # phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                # guess_ptr = ctypes.c_void_p(guess.ctypes.data)
                # total_ptr = ctypes.c_void_p(total_xs.ctypes.data)
                # source_ptr = ctypes.c_void_p(soures.ctypes.data)
                # maxi_ptr = ctypes.c_void_p(maxi.ctypes.data)
                # mini_ptr = ctypes.c_void_p(mini.ctypes.data)
                # ae_sweep(phi_ptr,guess_ptr,total_ptr,source_ptr,maxi_ptr,mini.ptr,ctypes.c_double(weight))
                # top_mult = (weight-half_total).astype('float64')
                # bottom_mult = (1/(weight+half_total)).astype('float64')
                # Left to right
                for ii in range(self.I):
                    # psi_top = eigen_eNDe.source_iteration(self,smult[ii],external[ii],psi_bottom,weight,guess[ii],ii)
                    
                    psi_top = eigen_eNDe.source_iteration(self,(external + smult)[ii],psi_bottom,weight,guess[ii],ii)
                    psi_top[psi_top < 0] = 0

                    verify = eigen_eNDe.angular_check(self,smult,external,psi_bottom,weight,ii)
                    # assert (np.sum(abs(psi_top - verify)) < 1e-5)
                    # if (np.sum(abs(psi_top - verify)) > 1e-5):
                        # print('Forward',ii,np.sum(abs(psi_top - verify)))
                    # psi_top = verify.copy()
                    
                    # ,normalize=[self.pmaxi[ii],self.pmini[ii]]
                    # psi_top = nnets.phi_normalize_single(psi_top,self.pmaxi[ii],self.pmini[ii])
                    # psi_top = eigen_eNDe.source_iteration(self,eigen_eNDe.decoding(self,smult[ii],'smult',ii),eigen_eNDe.decoding(self,external[ii],'fmult',ii),psi_bottom,weight,guess[ii],ii)
                    

                    # Q = smult[ii]+external[ii] + weight*psi_bottom - 0.5*self.total[ii]*psi_bottom
                    # xs = self.total[ii].copy()
                    # moo = weight.copy()
                    # psi_top = op.fixed_point(eigen_eNDe.function,np.zeros((87)),args=(Q,xs,moo),xtol=1e-10,maxiter=10000)
                    
                    phi[ii] = phi[ii] + (self.w[n] * 0.5 * (nnets.phi_normalize_single(psi_top,self.pmaxi[ii],self.pmini[ii]) + nnets.phi_normalize_single(psi_bottom,self.pmaxi[ii],self.pmini[ii])))
                    # * func.diamond_diff(psi_top,psi_bottom)
                    psi_bottom = psi_top.copy()
                for ii in range(self.I-1,-1,-1):
                    psi_top = psi_bottom.copy()

                    # psi_bottom = eigen_eNDe.source_iteration(self,smult[ii],external[ii],psi_top,weight,guess[ii],ii)

                    # psi_top = nnets.phi_normalize_single(psi_top,self.pmaxi[ii],self.pmini[ii])
                    # psi_bottom = eigen_eNDe.source_iteration(self,eigen_eNDe.decoding(self,smult[ii],'smult',ii),eigen_eNDe.decoding(self,external[ii],'fmult',ii),psi_top,weight,guess[ii],ii)
                    psi_bottom = eigen_eNDe.source_iteration(self,(external + smult)[ii],psi_top,weight,guess[ii],ii)
                    psi_bottom[psi_bottom < 0] = 0

                    verify = eigen_eNDe.angular_check(self,smult,external,psi_top,weight,ii)
                    # assert (np.sum(abs(psi_bottom - verify)) < 1e-5)
                    # if (np.sum(abs(psi_bottom - verify)) > 1e-5):
                        # print('Backward',ii,np.sum(abs(psi_bottom - verify)))
                    # psi_bottom = verify.copy()

                    # Q = smult[ii]+external[ii] + weight*psi_top - 0.5*self.total[ii]*psi_top
                    # xs = self.total[ii].copy()
                    # moo = weight.copy()
                    # psi_bottom = op.fixed_point(eigen_eNDe.function,np.zeros((87)),args=(Q,xs,moo),xtol=1e-10,maxiter=10000)

                    phi[ii] = phi[ii] + (self.w[n] * 0.5 * (nnets.phi_normalize_single(psi_top,self.pmaxi[ii],self.pmini[ii]) + nnets.phi_normalize_single(psi_bottom,self.pmaxi[ii],self.pmini[ii])))
                # phi = nnets.phi_normalize(phi,self.pmaxi,self.pmini)
                phi[np.isnan(phi)] = 0
                # print(np.sum(phi), np.isnan(phi).sum(), np.isinf(phi).sum())
            # phi = nnets.phi_normalize(phi,self.pmaxi,self.pmini)
            print(np.sum(phi), np.isnan(phi).sum(), np.isinf(phi).sum())
            phi += 1e-25
            change = np.linalg.norm((phi-phi_old)/phi/(self.I))
            phi -= 1e-25
            print('Change is',change,'count is',count)
            converged = (change < 1e-10) or (count >= 100) 
            count += 1
            # Calculate Sigma_s * phi
            phi[np.isnan(phi)] = 0
            phi_full = eigen_eNDe.decoding(self,phi,atype='phi')
            # _,self.pmaxi,self.pmini = nnets.normalize(phi_full,verbose=True)
            # nphi,_,_ = nnets.normalize(phi_full,verbose=True)
            # phi_full = phi.copy()
            smult_full = np.einsum('ijk,ik->ij',self.scatter,phi_full)
            # smult_full = np.einsum('ijk,ik->ij',self.scatter,phi_full)
            smult = eigen_eNDe.encoding(self,smult_full,atype='smult')
            # Update to phi G
            phi_old = phi.copy()   
        # return nnets.phi_normalize(phi,self.pmaxi,self.pmini)
        return phi

    def angular_check(self,smult,fmult,bottom,weight,cell):
        from discrete1.util import nnets

        temp1 = nnets.unnormalize_single(smult[cell],self.pmaxi[cell],self.pmini[cell])
        temp2 = nnets.unnormalize_single(fmult[cell],self.pmaxi[cell],self.pmini[cell])
        # print(np.sum(temp1+temp2))
        psi_top = (temp1 + temp2 + bottom *(weight - 0.5*self.total[cell]))/(weight + 0.5*self.total[cell])
        return psi_top
    # def function(x,Q,total,weight):
    #     return (Q - 0.5*x*total)/weight


    # def source_iteration(self,mult,source,psi_bottom,weight,guess,cell):
    #     import numpy as np
    #     from discrete1.util import nnets

    #     old = guess[None,:].copy()

    #     converged = 0; count = 1
    #     alpha_bottom = eigen_eNDe.decode_angular_encode(self,psi_bottom,cell,count)

    #     # alpha_bottom2 = nnets.unnormalize_single(alpha_bottom,self.tmaxi[cell],self.tmini[cell])
    #     # angular_up = nnets.phi_normalize_single(weight*psi_bottom,self.pmaxi[cell],self.pmini[cell])
    #     # angular_up[np.isnan(angular_up)] = 0; #angular_up[np.isinf(angular_up)] = 10
        
    #     # mult2 = nnets.unnormalize_single(mult,self.smaxi[cell],self.smini[cell])
    #     # source2 = nnets.unnormalize_single(source,self.fmaxi[cell],self.fmini[cell])
    #     # alpha_bottom = self.total[cell]*psi_bottom
    #     new = np.zeros((1,self.G))
    #     while not (converged):
    #         alpha_top = eigen_eNDe.decode_angular_encode(self,old,cell,count)
    #         # alpha_top = self.total[cell]*old
    #         # alpha_top2 = nnets.unnormalize_single(alpha_top,self.tmaxi[cell],self.tmini[cell])
    #         denominator = (old*weight+alpha_top)

    #         # angular_down = nnets.phi_normalize_single(old*weight,self.pmaxi[cell],self.pmini[cell])
    #         # angular_down[np.isnan(angular_down)] = 0; #angular_down[np.isinf(angular_down)] = 10
    #         # denominator = (angular_down+alpha_top)#-2*self.pmini[cell]

    #         # old = nnets.phi_normalize_single(old,self.pmaxi[cell],self.pmini[cell])
    #         # new = old*(mult+source+weight*psi_bottom-0.5*alpha_bottom)/denominator
    #         if np.argwhere(denominator == 0).shape[0] > 0:
    #             ind = np.argwhere(denominator != 0)[:,1].flatten()
    #             new = np.zeros((old.shape))
    #             # new[:,ind] = (old*(mult+source+weight*psi_bottom-alpha_bottom))[:,ind]/denominator[:,ind]
    #             new[:,ind] = (old*(mult+source+weight*psi_bottom-alpha_bottom))[:,ind]/denominator[:,ind]
    #         else:    
    #             # new = old*(mult+source+weight*psi_bottom-alpha_bottom)/denominator
    #             new = old*((mult+source+weight*psi_bottom-alpha_bottom)/denominator)

    #         new[np.isnan(new)] = 0; #new[np.isinf(new)] = 10

    #         #new[new < -0.25] = 0; #new[new > 10] = 1

    #         change = np.argwhere(abs(old-new) < 1e-12)
    #         converged = (len(change) == self.gprime) or (count >= 5000)

    #         old = new.copy(); count += 1

    #     return new # of size (1 x G_hat)

    def source_iteration(self,sources,psi_bottom,weight,guess,cell): #normalize=False
        import numpy as np
        from discrete1.util import nnets

        old = guess[None,:].copy()
        # alpha_bottom = 0.5*psi_bottom*total_xs
        alpha_bottom = eigen_eNDe.decode_angular_encode(self,psi_bottom,cell,1)
        # if normalize:
            # alpha_bottom = nnets.phi_normalize_single(alpha_bottom,normalize[0],normalize[1])
        converged = 0; count = 1
        new = np.zeros((1,self.gprime))
        while not (converged):
            # alpha_top = 0.5*total_xs*old
            alpha_top = eigen_eNDe.decode_angular_encode(self,old,cell,count)
            # if normalize:
            #     # alpha_top = nnets.phi_normalize_single(alpha_top,normalize[0],normalize[1])
            #     beta_top = nnets.phi_normalize_single(weight*psi_bottom,normalize[0],normalize[1])
            #     denominator = nnets.phi_normalize_single(old*weight,normalize[0],normalize[1])+alpha_top
            # else:
            #     beta_top = weight*psi_bottom
            #     denominator = (old*weight+alpha_top)
            beta_top = nnets.phi_normalize_single(weight*psi_bottom,self.pmaxi[cell],self.pmini[cell])
            denominator = nnets.phi_normalize_single(old*weight,self.pmaxi[cell],self.pmini[cell])+alpha_top
            if np.argwhere(denominator == 0).shape[0] > 0:
                ind = np.argwhere(denominator != 0)[:,1].flatten()
                new = np.zeros((old.shape))
                new[:,ind] = (old * (sources + beta_top - alpha_bottom))[:,ind]/denominator[:,ind]
            else:    
                new = old * ((sources + beta_top - alpha_bottom)/denominator)
            new[np.isnan(new)] = 0; #new[np.isinf(new)] = 10
            change = np.argwhere(abs(old-new) < 1e-14)
            converged = (len(change) == self.gprime) or (count >= 5000)
            old = new.copy(); count += 1

        return new 


    def decode_angular_encode(self,flux,ii,iteration):
        import numpy as np
        from discrete1.util import nnets
        # Decode
        flux_full = eigen_eNDe.decoding(self,flux,atype='phi',cell=ii)
        # Psi * Sigma_T
        mult_full = 0.5*self.total[ii]*flux
        mult_full[np.isnan(mult_full)] = 0
        # Encode
        mult = eigen_eNDe.encoding(self,mult_full,atype='phi',cell=ii)

        # mult = 0.5*self.total[ii]*flux

        return mult

    def encoding(self,matrix,atype,cell=None,normalize=True):
        import numpy as np
        from discrete1.util import nnets
        if atype == 'fmult':
            model = self.fmult_encoder
        elif atype == 'smult':
            model = self.smult_encoder
        elif atype == 'phi':
            model = self.phi_encoder

        if normalize:
            if cell is not None:
                matrix = nnets.phi_normalize_single(matrix,self.pmaxi[cell],self.pmini[cell])
            else:
                matrix = nnets.phi_normalize(matrix,self.pmaxi,self.pmini)
            # if atype == 'test_fmult':
            #     if cell is not None:
            #         matrix = nnets.phi_normalize_single(matrix,self.fmaxi[cell],self.fmini[cell])
            #     else:
            #         matrix = nnets.phi_normalize(matrix,self.fmaxi,self.fmini)
            # else:
            #     if cell is not None:
            #         matrix = nnets.phi_normalize_single(matrix,self.pmaxi[cell],self.pmini[cell])
            #     else:
            #         matrix = nnets.phi_normalize(matrix,self.pmaxi,self.pmini)
            matrix[np.isnan(matrix)] = 0; 

            # if cell is not None:
                # matrix = nnets.phi_normalize_single(matrix,self.pmaxi[cell],self.pmini[cell])
            # else:
                # matrix = nnets.phi_normalize(matrix,self.pmaxi,self.pmini)
                # matrix,self.pmaxi,self.pmini = nnets.normalize(matrix,verbose=True)
                # self.pmaxi[np.isnan(self.pmaxi)] = 0; self.pmini[np.isnan(self.pmini)] = 0; 

        # if atype == 'fmult':
        #     matrix,self.fmaxi,self.fmini = nnets.normalize(matrix,verbose=True)
        #     self.fmaxi[np.isnan(self.fmaxi)] = 0; self.fmini[np.isnan(self.fmini)] = 0; 
        #     # matrix = nnets.phi_normalize(matrix,self.pmaxi,self.pmini)
        # elif atype == 'smult':
        #     matrix,self.smaxi,self.smini = nnets.normalize(matrix,verbose=True)
        #     self.smaxi[np.isnan(self.smaxi)] = 0; self.smini[np.isnan(self.smini)] = 0; 
        #     # matrix = nnets.phi_normalize(matrix,self.pmaxi,self.pmini)
        # elif atype == 'phi':
        #     if cell is not None:
        #         # matrix,self.pmaxi[cell],self.pmini[cell] = nnets.normalize(matrix,verbose=True)
        #         matrix = nnets.normalize_single(matrix,verbose=False)
        #     else:
        #         matrix,self.pmaxi,self.pmini = nnets.normalize(matrix,verbose=True)
        #         # matrix = nnets.normalize(matrix)
        #     self.pmaxi[np.isnan(self.pmaxi)] = 0; self.pmini[np.isnan(self.pmini)] = 0; 

        matrix[np.isnan(matrix)] = 0; 
        # Scaling
        scale = np.sum(matrix,axis=1)
        matrix = model.predict(matrix)
        # matrix = (scale/np.sum(matrix,axis=1))[:,None]*matrix
        matrix = nnets.scale_back(scale,matrix)
        matrix[np.isnan(matrix)] = 0;

        return matrix

    def decoding(self,matrix,atype,cell=None,normalize=True):
        import numpy as np
        from discrete1.util import nnets
        if atype == 'fmult':
            model = self.fmult_decoder
        elif atype == 'smult':
            model = self.smult_decoder
        elif atype == 'phi':
            model = self.phi_decoder
 
        matrix[np.isnan(matrix)] = 0; 
        # Scaling
        scale = np.sum(matrix,axis=1)
        matrix = model.predict(matrix)
        # matrix = (scale/np.sum(matrix,axis=1))[:,None]*matrix
        matrix = nnets.scale_back(scale,matrix)
        matrix[np.isnan(matrix)] = 0;

        if normalize:
            if cell is not None:
                matrix = nnets.unnormalize_single(matrix,self.pmaxi[cell],self.pmini[cell])
            else:
                matrix = nnets.unnormalize(matrix,self.pmaxi,self.pmini)
            # if atype == 'test_fmult':
            #     if cell is not None:
            #         matrix = nnets.unnormalize_single(matrix,self.fmaxi[cell],self.fmini[cell])
            #     else:
            #         matrix = nnets.unnormalize(matrix,self.fmaxi,self.fmini)
            # else:
            #     if cell is not None:
            #         matrix = nnets.unnormalize_single(matrix,self.pmaxi[cell],self.pmini[cell])
            #     else:
            #         matrix = nnets.unnormalize(matrix,self.pmaxi,self.pmini)
            matrix[np.isnan(matrix)] = 0; 

        # # Unnormalize
        # if atype == 'fmult':
        #     if cell is not None:
        #         matrix = nnets.unnormalize_single(matrix,self.fmaxi[cell],self.fmini[cell])
        #     else:
        #         matrix = nnets.unnormalize(matrix,self.fmaxi,self.fmini)
            
        # elif atype == 'smult':
        #     if cell is not None:
        #         matrix = nnets.unnormalize_single(matrix,self.smaxi[cell],self.smini[cell])
        #     else:
        #         matrix = nnets.unnormalize(matrix,self.smaxi,self.smini)
            
        # elif atype == 'phi':
        #     if cell is not None:
        #         matrix = nnets.unnormalize_single(matrix,self.pmaxi[cell],self.pmini[cell])
        #     else:
        #         matrix = nnets.unnormalize(matrix,self.pmaxi,self.pmini)
        # matrix[np.isnan(matrix)] = 0;

        return matrix
            
    def transport(self,coder,problem='carbon',tol=1e-12,MAX_ITS=100):
        """ Arguments:
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a I x G array    """        
        import numpy as np
        from discrete1.util import nnets
        from discrete1.setup import func

        # Load Encoders, Decoders
        phi_autoencoder,phi_encoder,phi_decoder = func.load_coder(coder)
        self.phi_decoder = phi_decoder; self.phi_encoder = phi_encoder
        # Scatter
        smult_autoencoder,smult_encoder,smult_decoder = func.load_coder(coder,ptype='smult')
        self.smult_decoder = smult_decoder; self.smult_encoder = smult_encoder
        # Fission
        fmult_autoencoder,fmult_encoder,fmult_decoder = func.load_coder(coder,ptype='fmult')
        self.fmult_decoder = fmult_decoder; self.fmult_encoder = fmult_encoder
        # Set Encoded Layer Dimension - need function
        self.gprime = 87

        # Initialize flux
        phi_old_full = func.initial_flux(problem)
        _,self.pmaxi,self.pmini = nnets.normalize(phi_old_full,verbose=True)
        
        # Initialize source
        sources_full = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old_full)
        # _,self.fmaxi,self.fmini = nnets.normalize(sources_full,verbose=True)
        smult_full = np.einsum('ijk,ik->ij',self.scatter,phi_old_full)
         # _,self.smaxi,self.smini = nnets.normalize(smult_full,verbose=True)
        
        # Encode Current variables
        phi_old = eigen_eNDe.encoding(self,phi_old_full,atype='phi')
        sources = eigen_eNDe.encoding(self,sources_full,atype='fmult')
        smult = eigen_eNDe.encoding(self,smult_full,atype='smult')

        # assert (np.array_equal(phi_old,phi_old_full))
        # assert (np.array_equal(sources,sources_full))
        converged = 0; count = 1
        while not (converged):
            print('Outer Transport Iteration {}\n==================================='.format(count))
            # Calculate phi G'
            phi = eigen_eNDe.multi_group(self,smult,sources,phi_old)
            # Convert to phi G, normalized
            # print('Transport Code')
            # print(np.sum(phi), np.isnan(phi).sum(), np.isinf(phi).sum())
            phi_full = eigen_eNDe.decoding(self,phi,atype='phi')
            # phi_full = phi.copy()
            # print(np.sum(phi_full), np.isnan(phi_full).sum(), np.isinf(phi_full).sum())
            phi_full[np.isnan(phi_full)] = 0
            # print(np.array_equal(phi,phi_full))
            keff = np.linalg.norm(phi_full)
            phi_full /= keff


            # Check for convergence with original phi sizes            
            # temp1 = (phi_full-phi_old_full)/phi_full/(self.I)
            # temp1[np.isnan(temp1)] = 0; temp1[np.isinf(temp1)] = 0
            # change = np.linalg.norm(temp1)
            phi_full += 1e-25
            change = np.linalg.norm((phi_full-phi_old_full)/phi_full/(self.I))
            phi_full -= 1e-25
            
            phi_old_full[np.isnan(phi_old_full)] = 0; phi_old_full[np.isinf(phi_old_full)] = 0
            phi_full[np.isnan(phi_full)] = 0; phi_full[np.isinf(phi_full)] = 0
            print('Change is',change,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS)
            count += 1

            # Update to phi G
            phi_old_full = phi_full.copy()
            _,self.pmaxi,self.pmini = nnets.normalize(phi_old_full,verbose=True)
            # Recalculate Sources
            sources_full = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old_full)
            # _,self.fmaxi,self.fmini = nnets.normalize(sources_full,verbose=True)
            smult_full = np.einsum('ijk,ik->ij',self.scatter,phi_old_full) #phi_old_full
            # _,self.smaxi,self.smini = nnets.normalize(smult_full,verbose=True)
            # Encode back down
            phi_old = eigen_eNDe.encoding(self,phi_old_full,atype='phi')
            # assert np.array_equal(phi_full,phi_old)
            sources = eigen_eNDe.encoding(self,sources_full,atype='fmult')
            smult = eigen_eNDe.encoding(self,smult_full,atype='smult')

        return phi_full,keff

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
        if self.multAE == 'smult' or self.multAE == 'both':
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
            # _,self.pmaxi,self.pmini = nnets.normalize(phi_old_full,verbose=True)
            phi_old = eigen_auto.scale_autoencode(self,phi_old_full,atype='phi')
            # phi_old = phi.copy()
            # Encode Scatter * Phi
            smult_full = np.einsum('ijk,ik->ij',scatter,phi_full)
            if self.multAE == 'smult' or self.multAE == 'both':
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
            # models = [self.phi_autoencoder1,self.phi_autoencoder2,self.phi_autoencoder3]
        matrix,self.pmaxi,self.pmini = nnets.normalize(matrix_full,verbose=True)

        # matrix = nnets.phi_normalize(matrix_full,self.pmaxi,self.pmini)
        matrix[np.isnan(matrix)] = 0; self.pmaxi[np.isnan(self.pmaxi)] = 0; self.pmini[np.isnan(self.pmini)] = 0
        scale = np.sum(matrix,axis=1)

        # parts = matrix.copy()
        # matrix = np.empty(parts.shape)
        # for ind,ae in zip(self.splits,models):
        #     matrix[ind] = ae.predict(parts[ind])

        matrix = model.predict(matrix)
        matrix = (scale/np.sum(matrix,axis=1))[:,None]*matrix
        matrix[np.isnan(matrix)] = 0;
        matrix = nnets.unnormalize(matrix,self.pmaxi,self.pmini)
    
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
        from discrete1.setup import func
        # from discrete1.util import sn
        phi_autoencoder,phi_encoder,phi_decoder = func.load_coder(coder)
        self.phi_autoencoder = phi_autoencoder

        # ae_model1 = 'eigen_nn/epochs_250/model40-20_hdpe'
        # ae_model2 = 'eigen_nn/epochs_250/model40-20_u235'
        # ae_model3 = 'eigen_nn/epochs_250/model40-20_u238'

        # self.phi_autoencoder1,_,_ = func.load_coder(ae_model1)
        # self.phi_autoencoder2,_,_ = func.load_coder(ae_model2)
        # self.phi_autoencoder3,_,_ = func.load_coder(ae_model3)
        # self.splits = [slice(0,450),slice(450,800),slice(800,1000)]

        self.multAE = multAE
        # print('========================')
        # print(multAE)
        # print('========================')
        self.gprime = 87

        if self.multAE == 'smult' or self.multAE == 'both':
            smult_autoencoder,smult_encoder,smult_decoder = func.load_coder(coder,ptype='smult')
            self.smult_autoencoder = smult_autoencoder
        if self.multAE == 'fmult' or self.multAE == 'both':
            fmult_autoencoder,fmult_encoder,fmult_decoder = func.load_coder(coder,ptype='fmult')
            self.fmult_autoencoder = fmult_autoencoder

        phi_old_full = func.initial_flux(problem)
        # _,self.pmaxi,self.pmini = nnets.normalize(phi_old_full,verbose=True)
        keff = np.linalg.norm(phi_old_full)

        sources_full = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old_full)
        # Encode-Decode Phi
        phi_old = eigen_auto.scale_autoencode(self,phi_old_full,atype='phi')
        # Unnormalized Method
        # phi_old = self.phi_autoencoder.predict(phi_old_full)
        # phi_old = (np.sum(phi_old_full,axis=1)/np.sum(phi_old,axis=1))[:,None]*phi_old
        if self.multAE == 'fmult' or self.multAE == 'both':
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
            # phi_old = phi.copy()
            # _,self.pmaxi,self.pmini = nnets.normalize(phi_old_full,verbose=True)

            sources_full = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old_full)
            if self.multAE == 'fmult' or self.multAE == 'both':
                sources = eigen_auto.scale_autoencode(self,sources_full,atype='fmult')
            else:
                sources = sources_full.copy()
            phi_old = eigen_auto.scale_autoencode(self,phi_old_full,atype='phi')

        return phi_full,keff

class eigen_auto_djinn:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L,R,I,enrich=None,splits=None,track=None,label=None):
        self.G = G
        self.N = N
        self.mu = mu
        self.w = w
        self.total = total
        self.scatter = scatter
        self.chiNuFission = chiNuFission
        self.I = I
        self.inv_delta = float(I)/R
        self.enrich = enrich
        self.splits = splits
        self.track = track
        self.label = label
        
    def one_group(self,total,scatter,djinn_1g,external,guess,tol=1e-08,MAX_ITS=100):
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
        from discrete1.util import sn
        import ctypes
        clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cfunctions.so')
        sweep = clibrary.sweep
        # converged = 0
        # count = 1        
        phi = np.zeros((self.I),dtype='float64')
        phi_old = guess.copy()
        half_total = 0.5*total.copy()
        external = external.astype('float64')
        for n in range(self.N):
            weight = self.mu[n]*self.inv_delta
            top_mult = (weight-half_total).astype('float64')
            bottom_mult = (1/(weight+half_total)).astype('float64')
            if (self.multAE == 'scatter' or self.multAE == 'both'):
                temp_scat = sn.pops_robust('scatter',(self.I,),sn.cat(scatter*phi_old,self.splits['scatter_keep']),djinn_1g,self.splits).astype('float64')
            else:
                temp_scat = (scatter * phi_old).astype('float64')
            # Set Pointers for C function
            phi_ptr = ctypes.c_void_p(phi.ctypes.data)
            ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)
            ext_ptr = ctypes.c_void_p(external.ctypes.data)
            top_ptr = ctypes.c_void_p(top_mult.ctypes.data)
            bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)
            sweep(phi_ptr,ts_ptr,ext_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]))

        return phi

    def multi_group(self,total,scatter,nuChiFission,guess,tol=1e-08,MAX_ITS=10000):
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
        from discrete1.setup import func
        
        phi_old = guess.copy()
        
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)
            if self.multAE == 'scatter' or self.multAE == 'both':
                smult = eigen_auto_djinn.scale_autoencode(self,phi_old,atype='smult')
                for g in range(self.G):
                    phi[:,g] = eigen_auto_djinn.one_group(self,total[:,g],scatter[:,g,g],smult[:,g],nuChiFission[:,g],phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
            elif self.multAE == 'fission':
                for g in range(self.G):
                    if g == 0:
                        q_tilde = nuChiFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g)
                    else:
                        q_tilde = nuChiFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g) + func.update_q(scatter,phi,0,g,g)
                    phi[:,g] = eigen_auto_djinn.one_group(self,total[:,g],scatter[:,g,g],None,q_tilde,phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)

            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 
            phi_old = phi.copy()

        return phi
    
    def scale_autoencode(self,flux,atype):
        import numpy as np
        from discrete1.util import nnets

        matrix,maxi,mini = nnets.normalize(flux,verbose=True)

        matrix[np.isnan(matrix)] = 0; maxi[np.isnan(maxi)] = 0; mini[np.isnan(mini)] = 0
        scale = np.sum(matrix,axis=1)

        encoded_flux = self.phi_encoder.predict(matrix)
        encoded_flux = (scale/np.sum(encoded_flux,axis=1))[:,None]*encoded_flux
        # print(encoded_flux.shape)
        if atype == 'fmult':
            sources = eigen_auto_djinn.create_fmult(self,encoded_flux,flux)
            fmult_scale = np.sum(sources,axis=1)
            decoded_source = self.fmult_decoder.predict(sources)
            decoded_source = (fmult_scale/np.sum(decoded_source,axis=1))[:,None]*decoded_source
            decoded_source[np.isnan(decoded_source)] = 0;
            return nnets.unnormalize(decoded_source,maxi,mini)

        elif atype == 'smult':
            smult = eigen_auto_djinn.create_smult(self,encoded_flux,flux)
            smult_scale = np.sum(smult,axis=1)
            decoded_smult = self.smult_decoder.predict(smult)
            decoded_smult = (smult_scale/np.sum(decoded_smult,axis=1))[:,None]*decoded_smult
            decoded_smult[np.isnan(decode_smult)] = 0;
            return nnets.unnormalize(decoded_smult,maxi,mini)

        else:
            return "Do not understand data type"

    def label_model(self,xs,flux,model_):
        import numpy as np
        from discrete1.util import sn
        phi = flux.copy()
        if np.sum(phi) == 0:
            return np.zeros((sn.cat(phi,self.splits['{}_djinn'.format(xs)]).shape))
        if xs == 'scatter':
            nphi = np.linalg.norm(phi)
            phi /= nphi
        short_phi = sn.cat(phi,self.splits['{}_djinn'.format(xs)])
        # if self.process == 'norm':
        #     short_phi /= np.linalg.norm(short_phi,axis=1)[:,None]
        if self.label:
            short_phi = np.hstack((sn.cat(self.enrich,self.splits['{}_djinn'.format(xs)])[:,None],short_phi))
        # if xs == 'scatter':
        #     return model_.predict(short_phi),nphi
        return model_.predict(short_phi) 

    def scale_scatter(self,phi,phi_full,djinn_ns):
        import numpy as np
        from discrete1.util import sn
        if np.sum(phi) == 0:
            return np.zeros((sn.cat(phi,self.splits['scatter_djinn']).shape))
        interest = sn.cat(phi_full,self.splits['scatter_djinn'])
        scale = np.sum(interest*np.sum(sn.cat(self.scatter,self.splits['scatter_djinn']),axis=1),axis=1)/np.sum(djinn_ns,axis=1)
        return scale[:,None]*djinn_ns

    def create_smult(self,flux,flux_full):
        import numpy as np
        if (np.sum(flux) == 0):
            return np.zeros(flux.shape)
        djinn_scatter_ns = eigen_auto_djinn.label_model(self,'scatter',flux,self.dj_scatter)
        return eigen_auto_djinn.scale_scatter(self,flux,flux_full,djinn_scatter_ns)#*nphi

    def scale_fission(self,phi,phi_full,djinn_ns):
        import numpy as np
        from discrete1.util import sn
        if np.sum(phi) == 0:
            return np.zeros((sn.cat(phi,self.splits['fission_djinn']).shape))
        if self.multAE == 'scatter':
            return np.einsum('ijk,ik->ij',self.chiNuFission,phi) 
        interest = sn.cat(phi_full,self.splits['fission_djinn'])
        scale = np.sum(interest*np.sum(sn.cat(self.chiNuFission,self.splits['fission_djinn']),axis=1),axis=1)/np.sum(djinn_ns,axis=1)
        # All of the sigma*phi terms not calculated by DJINN
        regular = np.einsum('ijk,ik->ij',sn.cat(self.chiNuFission,self.splits['fission_keep']),sn.cat(phi_full,self.splits['fission_keep']))
        # print('regular shape',regular.shape)
        # print(phi.shape,regular[:,20])
        regular = regular[:,:20].copy()
        return sn.pops_robust('fission',phi.shape,regular,scale[:,None]*djinn_ns,self.splits)
        
    def create_fmult(self,flux,flux_full):
        djinn_fission_ns = 0
        if self.multAE == 'both' or self.multAE == 'fission':
            djinn_fission_ns = eigen_auto_djinn.label_model(self,'fission',flux,self.dj_fission)
        return eigen_auto_djinn.scale_fission(self,flux,flux_full,djinn_fission_ns)

    def transport(self,ae_name,dj_name,problem='carbon',tol=1e-12,MAX_ITS=100,LOUD=True,multAE=False):
        """ Arguments:
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a I x G array    """        
        import numpy as np
        from discrete1.util import nnets
        from discrete1.setup import func
        # from discrete1.util import sn
        phi_autoencoder,phi_encoder,phi_decoder = func.load_coder(ae_name)
        self.phi_encoder = phi_encoder
        self.multAE = multAE

        self.dj_scatter,self.dj_fission = func.djinn_load(dj_name,self.multAE)
        if self.multAE == 'scatter' or self.multAE == 'both':
            smult_autoencoder,smult_encoder,smult_decoder = func.load_coder(ae_name,ptype='smult')
            self.smult_decoder = smult_decoder
        if self.multAE == 'fission' or self.multAE == 'both':
            fmult_autoencoder,fmult_encoder,fmult_decoder = func.load_coder(ae_name,ptype='fmult')
            self.fmult_decoder = fmult_decoder

        phi_old = func.initial_flux(problem)

        if self.multAE == 'fission' or self.multAE == 'both':
            sources = eigen_auto_djinn.scale_autoencode(self,phi_old,atype='fmult')
        else:
            sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old)

        converged = 0; count = 1
        while not (converged):
            print('Outer Transport Iteration {}'.format(count))
            phi = eigen_auto_djinn.multi_group(self,self.total,self.scatter,sources,phi_old,tol=1e-08,MAX_ITS=MAX_ITS)
            
            keff = np.linalg.norm(phi)
            phi /= keff

            change = np.linalg.norm((phi-phi_old)/phi/(self.I))
            # change = np.linalg.norm((phi-phi_old)/phi/(self.I))
            if LOUD:
                print('Change is',change,'Keff is',keff)
                print('===================================')
            converged = (change < tol) or (count >= MAX_ITS) #or (kchange < tol)
            count += 1

            phi_old = phi.copy()
        
            if self.multAE == 'fission' or self.multAE == 'both':
                sources = eigen_auto_djinn.scale_autoencode(self,phi_old,atype='fmult')
            else:
                sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old)

        return phi,keff


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
        from discrete1.setup import ex_sources,func
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
            
class source_eNDe:
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

    def multi_group(self,smult,external,guess):
        import numpy as np
        from discrete1.util import nnets
        from discrete1.setup import func
        import ctypes

        # cfunctions = ctypes.cdll.LoadLibrary('./discrete1/data/clibrary.so')
        # ae_sweep = clibrary.ae_sweep
        guess = guess.astype('float64')
        # sources = (smult + external).astype('float64')
        total_xs = self.total.astype('float64')
        # maxi = (self.pmaxi).astype('float64')
        # mini = (self.pmini).astype('float64')
        half_total = 0.5*self.total.copy()
        phi = np.zeros((self.I,self.gprime),dtype='float64')
        for n in range(self.N):
            weight = (self.mu[n]*self.inv_delta).astype('float64')
            psi_bottom = np.zeros((1,self.gprime))
            # psi_bottom = np.zeros((1,self.gprime)) # vacuum on LHS
            # phi_ptr = ctypes.c_void_p(phi.ctypes.data)
            # guess_ptr = ctypes.c_void_p(guess.ctypes.data)
            # total_ptr = ctypes.c_void_p(total_xs.ctypes.data)
            # source_ptr = ctypes.c_void_p(soures.ctypes.data)
            # maxi_ptr = ctypes.c_void_p(maxi.ctypes.data)
            # mini_ptr = ctypes.c_void_p(mini.ctypes.data)
            # ae_sweep(phi_ptr,guess_ptr,total_ptr,source_ptr,maxi_ptr,mini.ptr,ctypes.c_double(weight))
            
            top_mult = (weight-half_total).astype('float64')
            bottom_mult = (1/(weight+half_total)).astype('float64')
                        
            # Left to right
            for ii in range(self.I):
                psi_top = source_eNDe.source_iteration(self,smult[ii],external[ii],psi_bottom,weight,guess[ii],ii)
                phi[ii] = phi[ii] + (self.w[n] * func.diamond_diff(psi_top,psi_bottom))
                psi_bottom = psi_top.copy()
            for ii in range(self.I-1,-1,-1):
                psi_top = psi_bottom.copy()
                psi_bottom = source_eNDe.source_iteration(self,smult[ii],external[ii],psi_top,weight,guess[ii],ii)
                phi[ii] = phi[ii] +  (self.w[n] * func.diamond_diff(psi_top,psi_bottom))
        return phi


    def source_iteration(self,mult,source,psi_bottom,weight,guess,cell):
        import numpy as np
        old = guess[None,:].copy()
        converged = 0; count = 1
        alpha_bottom = source_eNDe.decode_angular_encode(self,psi_bottom,cell)
        new = np.zeros((1,self.G))
        while not (converged):
            alpha_top = source_eNDe.decode_angular_encode(self,old,cell)
            new = old*(mult+source+weight*psi_bottom-0.5*alpha_bottom)/(old*weight+0.5*alpha_top)
            new[new < -0.5] = 0
            new[np.isnan(new)] = 0
            change = np.argwhere(abs(old-new) < 1e-8)
            converged = (len(change) == 87) or (count >= 500)
            old = new.copy(); count += 1
        return new # of size (1 x G_hat)


    def decode_angular_encode(self,flux,ii):
        import numpy as np
        from discrete1.util import nnets
        # Decode
        flux_full = source_eNDe.decoding(self,flux,atype='phi',cell=ii)
        # Psi * Sigma_T
        mult_full = self.total[ii]*flux_full
        # Encode
        mult = source_eNDe.encoding(self,mult_full,atype='phi',cell=ii)
        return mult

    def encoding(self,matrix,atype,cell=None):
        import numpy as np
        from discrete1.util import nnets
        if atype == 'fmult':
            model = self.fmult_encoder
            # matrix,self.fmaxi,self.fmini = nnets.normalize(matrix,verbose=True)
        elif atype == 'smult':
            model = self.smult_encoder
            # matrix,self.smaxi,self.smini = nnets.normalize(matrix,verbose=True)
        elif atype == 'phi':
            model = self.phi_encoder
            # if cell is not None:
            #     matrix,self.pmaxi[cell],self.pmini[cell] = nnets.normalize(matrix,verbose=True)
            # else:
            #     matrix,self.pmaxi,self.pmini = nnets.normalize(matrix,verbose=True)
        # matrix[np.isnan(matrix)] = 0; 
        # Scaling
        scale = np.sum(matrix,axis=1)
        matrix = model.predict(matrix)
        matrix = (scale/np.sum(matrix,axis=1))[:,None]*matrix
        matrix[np.isnan(matrix)] = 0;
        # Unnormalize Here?
        # if atype == 'fmult':
        #     matrix = nnets.unnormalize(matrix,self.fmaxi,self.fmini)
        # elif atype == 'smult':
        #     matrix = nnets.unnormalize(matrix,self.smaxi,self.smini)
        # elif atype == 'phi':
        #     if cell is not None:
        #         matrix = nnets.unnormalize_single(matrix,self.pmaxi[cell],self.pmini[cell])
        #     else:
        #         matrix = nnets.unnormalize(matrix,self.pmaxi,self.pmini)
        # matrix[np.isnan(matrix)] = 0;
        return matrix

    def decoding(self,matrix,atype,cell=None):
        import numpy as np
        from discrete1.util import nnets
        if atype == 'fmult':
            model = self.fmult_decoder
            # matrix,self.fmaxi,self.fmini = nnets.normalize(matrix_full,verbose=True)
        elif atype == 'smult':
            model = self.smult_decoder
            # matrix,self.smaxi,self.smini = nnets.normalize(matrix_full,verbose=True)
        elif atype == 'phi':
            model = self.phi_decoder
            # if cell is not None:
            #     matrix = nnets.unnormalize_single(matrix,self.pmaxi[cell],self.pmini[cell])
            # else:
            #     matrix = nnets.unnormalize(matrix,self.pmaxi,self.pmini)
            # matrix,self.pmaxi,self.pmini = nnets.normalize(matrix_full,verbose=True)
        # matrix[np.isnan(matrix)] = 0; 
        # Scaling
        scale = np.sum(matrix,axis=1)
        matrix = model.predict(matrix)
        matrix = (scale/np.sum(matrix,axis=1))[:,None]*matrix
        matrix[np.isnan(matrix)] = 0;
        # Normalize
        # if atype == 'fmult':
        #     matrix = nnets.unnormalize(matrix,self.fmaxi,self.fmini)
        # elif atype == 'smult':
        #     matrix = nnets.unnormalize(matrix,self.smaxi,self.smini)
        # elif atype == 'phi':
        #     if cell is not None:
        #         matrix = nnets.unnormalize_single(matrix,self.pmaxi[cell],self.pmini[cell])
        #     else:
        #         matrix = nnets.unnormalize(matrix,self.pmaxi,self.pmini)
        #     # matrix = nnets.unnormalize(matrix,self.pmaxi,self.pmini)
        # matrix[np.isnan(matrix)] = 0;
        return matrix



    def decode_angular_encode(self,flux,ii):
        import numpy as np
        from discrete1.util import nnets
        # Decode
        flux_full = source_eNDe.decoding(self,flux,atype='phi',cell=ii)
        # Psi * Sigma_T
        mult_full = self.total[ii]*flux_full
        # Encode
        mult = source_eNDe.encoding(self,mult_full,atype='phi',cell=ii)
        return mult

    def encoding(self,matrix,atype,cell=None):
        import numpy as np
        from discrete1.util import nnets
        if atype == 'fmult':
            model = self.fmult_encoder
            # matrix,self.fmaxi,self.fmini = nnets.normalize(matrix,verbose=True)
        elif atype == 'smult':
            model = self.smult_encoder
            # matrix,self.smaxi,self.smini = nnets.normalize(matrix,verbose=True)
        elif atype == 'phi':
            model = self.phi_encoder
            # if cell is not None:
            #     matrix,self.pmaxi[cell],self.pmini[cell] = nnets.normalize(matrix,verbose=True)
            # else:
            #     matrix,self.pmaxi,self.pmini = nnets.normalize(matrix,verbose=True)
        matrix[np.isnan(matrix)] = 0; 
        # Scaling
        scale = np.sum(matrix,axis=1)
        matrix = model.predict(matrix)
        matrix = (scale/np.sum(matrix,axis=1))[:,None]*matrix
        matrix[np.isnan(matrix)] = 0;
        # Unnormalize Here?
        # if atype == 'fmult':
        #     matrix = nnets.unnormalize(matrix,self.fmaxi,self.fmini)
        # elif atype == 'smult':
        #     matrix = nnets.unnormalize(matrix,self.smaxi,self.smini)
        # elif atype == 'phi':
        #     if cell is not None:
        #         matrix = nnets.unnormalize_single(matrix,self.pmaxi[cell],self.pmini[cell])
        #     else:
        #         matrix = nnets.unnormalize(matrix,self.pmaxi,self.pmini)
        # matrix[np.isnan(matrix)] = 0;
        return matrix

    def decoding(self,matrix,atype,cell=None):
        import numpy as np
        from discrete1.util import nnets
        if atype == 'fmult':
            model = self.fmult_decoder
            # matrix,self.fmaxi,self.fmini = nnets.normalize(matrix_full,verbose=True)
        elif atype == 'smult':
            model = self.smult_decoder
            # matrix,self.smaxi,self.smini = nnets.normalize(matrix_full,verbose=True)
        elif atype == 'phi':
            model = self.phi_decoder
            # if cell is not None:
            #     matrix = nnets.unnormalize_single(matrix,self.pmaxi[cell],self.pmini[cell])
            # else:
            #     matrix = nnets.unnormalize(matrix,self.pmaxi,self.pmini)
            # matrix,self.pmaxi,self.pmini = nnets.normalize(matrix_full,verbose=True)
        matrix[np.isnan(matrix)] = 0; 
        # Scaling
        scale = np.sum(matrix,axis=1)
        matrix = model.predict(matrix)
        matrix = (scale/np.sum(matrix,axis=1))[:,None]*matrix
        matrix[np.isnan(matrix)] = 0;
        # Normalize
        # if atype == 'fmult':
        #     matrix = nnets.unnormalize(matrix,self.fmaxi,self.fmini)
        # elif atype == 'smult':
        #     matrix = nnets.unnormalize(matrix,self.smaxi,self.smini)
        # elif atype == 'phi':
        #     if cell is not None:
        #         matrix = nnets.unnormalize_single(matrix,self.pmaxi[cell],self.pmini[cell])
        #     else:
        #         matrix = nnets.unnormalize(matrix,self.pmaxi,self.pmini)
        #     # matrix = nnets.unnormalize(matrix,self.pmaxi,self.pmini)
        # matrix[np.isnan(matrix)] = 0;
        return matrix


    def transport(self,coder,problem='carbon',tol=1e-08,MAX_ITS=10):
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
        self.gprime = 87

        # Initialize phi
        phi_old_full = func.initial_flux('carbon_source')
        

        # Initialize sources
        smult_full = np.einsum('ijk,ik->ij',self.scatter,phi_old_full)
        fmult_full = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old_full)
        print('Original Shapes',phi_old_full.shape,smult_full.shape,fmult_full.shape)

        # Encode Current Problems
        # Everything is normalized
        phi_old = source_eNDe.encoding(self,phi_old_full,atype='phi')
        smult = source_eNDe.encoding(self,smult_full,atype='smult')
        fmult = source_eNDe.encoding(self,fmult_full,atype='fmult')
        mult = smult + fmult
        print('Encoded Shapes',phi_old.shape,smult.shape,fmult.shape)
        # print(np.array_equal(phi_old_full,phi_old))
        # How Am I Suppose to encode this?
        source = ex_sources.source1(self.I,self.G)
        # source = self.phi_encoder.predict(source)

        converged = 0; count = 1
        while not (converged):
            print('Source Iteration {}'.format(count))
            phi = source_eNDe.multi_group(self,mult,source,phi_old)
            # Decode out phi
            phi_full = source_eNDe.decoding(self,phi,atype='phi')
            # phi_full = phi.copy()
            # Calculate change
            change = np.linalg.norm((phi_full - phi_old_full)/phi_full/(self.I))
            # Check for convergence
            print('Change is',change,'\n===================================')    
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 
            print(np.sum(phi))
            # Continue with the next iterations
            # print(np.sum(phi_full))
            phi_old_full = phi_full.copy()
            phi_old = source_eNDe.encoding(self,phi_old_full,atype='phi')
            # print(np.array_equal(phi_old_full,phi_old))
            # Calculate full mult matrices
            smult_full = np.einsum('ijk,ik->ij',self.scatter,phi_old_full)
            fmult_full = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old_full)
            # Encode down
            smult = source_eNDe.encoding(self,smult_full,atype='smult')
            fmult = source_eNDe.encoding(self,fmult_full,atype='fmult')
            mult = smult + fmult

        return phi_full
