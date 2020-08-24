class eigen:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L,R,I,track=False,enrich=None,splits=None):
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
        self.enrich = enrich
        self.splits = splits
        
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
    
    def multi_group(self,total,scatter,nuChiFission,tol=1e-08,MAX_ITS=100):
        """ Arguments:
            total: I x G vector of the total cross section for each spatial cell and energy level
            scatter: I x G array for the scattering of the spatial cell by moment and energy
            nuChiFission: 
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
        Returns:
            phi: a I x G array  """
        import numpy as np
        from discrete1.theProcess import func

        phi_old = np.zeros((self.I,self.G))
        converged = 0
        count = 1
        if self.track == 'source':
            allmat_sca = np.zeros((2,0,self.G+1))
        while not (converged):
            phi = np.zeros(phi_old.shape)
            for g in range(self.G):
                if g == 0:
                    q_tilde = nuChiFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g)
                else:
                    q_tilde = nuChiFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g) + func.update_q(scatter,phi,0,g,g)
                phi[:,g] = eigen.one_group(self,total[:,g],scatter[:,g,g],q_tilde,tol=tol,MAX_ITS=MAX_ITS,guess=phi_old[:,g])
            if self.track == "source":
                temp_fission,temp_scatter = eigen.tracking_data(self,phi,np.empty((1000,87)))
                allmat_sca = np.hstack((allmat_sca,temp_scatter))
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        if self.track == 'source':
            return phi,allmat_sca
        return phi
            
    def transport(self,problem,enrich,tol=1e-12,MAX_ITS=100,LOUD=True):
        """ Arguments:
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a I x G array    """        
        import numpy as np

        phi_old = np.random.rand(self.I,self.G)
        phi_old /= np.linalg.norm(phi_old)
        converged = 0
        count = 1
        if self.track == 'power':
            allmat_sca = np.zeros((2,0,self.G+1))
            allmat_fis = np.zeros((2,0,self.G+1))
        sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old) 
        if self.track == 'source':
            temp_fission2,temp_scatter2 = eigen.tracking_data(self,phi_old,sources)
            np.save('mydata/track_{}_djinn/enrich_{:<02}_count_000'.format(problem,enrich),temp_scatter2)
        while not (converged):
            if self.track == 'power':
                temp_fission,temp_scatter = eigen.tracking_data(self,phi_old,sources)
                allmat_sca = np.hstack((allmat_sca,temp_scatter))
                allmat_fis = np.hstack((allmat_fis,temp_fission))
            print('Outer Transport Iteration {}\n==================================='.format(count))
            if self.track == 'source':
                phi,temp_scatter2 = eigen.multi_group(self,self.total,self.scatter,sources,tol=1e-08,MAX_ITS=MAX_ITS)
                np.save('mydata/track_{}_djinn/enrich_{:<02}_count_{}'.format(problem,enrich,str(count).zfill(3)),temp_scatter2)
            else:
                phi = eigen.multi_group(self,self.total,self.scatter,sources,tol=1e-08,MAX_ITS=MAX_ITS)
            keff = np.linalg.norm(phi)
            phi /= keff
                        
            change = np.linalg.norm((phi-phi_old)/phi/(self.I))
            if LOUD:
                print('Change is',change,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1

            phi_old = phi.copy()
            sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old) 
        if self.track == 'power':
            return phi,keff,allmat_fis,allmat_sca
        return phi,keff

    def tracking_data(self,phi,sources=None):
        from discrete1.util import sn
        import numpy as np
        # Scatter Tracking - separate phi and add label
        label_scatter = sn.cat(self.enrich,self.splits['scatter_djinn'])
        phi_scatter = sn.cat(phi,self.splits['scatter_djinn'])
        phi_scatter /= np.linalg.norm(phi_scatter)
        phi_full_scatter = np.hstack((label_scatter[:,None],phi_scatter))
        # Separate scatter multiplier and add label
        multiplier_scatter = np.einsum('ijk,ik->ij',sn.cat(self.scatter,self.splits['scatter_djinn']),phi_scatter)
        multiplier_full_scatter = np.hstack((label_scatter[:,None],multiplier_scatter))
        scatter_data = np.vstack((phi_full_scatter[None,:,:],multiplier_full_scatter[None,:,:]))
        # Fission Tracking - Separate phi and add label
        label_fission = sn.cat(self.enrich,self.splits['fission_djinn'])
        phi_fission = sn.cat(phi,self.splits['fission_djinn'])
        phi_full_fission = np.hstack((label_fission[:,None],phi_fission))
        # Separate fission multiplier and add label
        multiplier_fission = sn.cat(sources,self.splits['fission_djinn'])
        multiplier_full_fission = np.hstack((label_fission[:,None],multiplier_fission))
        fission_data = np.vstack((phi_full_fission[None,:,:],multiplier_full_fission[None,:,:]))
        return fission_data, scatter_data

class eigen_collect:
    "The hokey way of doing Sn Transport"
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

        phi_old = guess.copy()
        smult = np.einsum('ijk,ik->ij',scatter,phi_old)

        converged = 0
        count = 1
        if self.track == 'source':
            allmat_phi = []
        while not (converged):
            phi = np.zeros(phi_old.shape)  
            for g in range(self.G):
                phi[:,g] = eigen_collect.one_group(self,total[:,g],smult[:,g],nuChiFission[:,g],phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
            if self.track == "source":
                allmat_phi.append(phi)
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 
            
            phi_old = phi.copy()
            smult = np.einsum('ijk,ik->ij',scatter,phi)
        if self.track == 'source':
            return phi,np.concatenate((allmat_phi))
        return phi
            
    def transport(self,enrich,problem='carbon',tol=1e-12,MAX_ITS=100,LOUD=True,normal=True):
        """ Arguments:
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a I x G array    """        
        import numpy as np
        from discrete1.util import nnets
        from discrete1.theProcess import func

        phi_old = func.initial_flux(problem)
        keff = np.linalg.norm(phi_old)
        sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old)
        if self.track == 'source':
            np.save('mydata/track_ae_{}/phi_{:<02}_count_000'.format(problem,enrich),phi_old)
        elif self.track == 'power':
            complete_phi = phi_old.copy()
        converged = 0
        count = 1
        while not (converged):
            print('Outer Transport Iteration {}'.format(count))
            if self.track == 'source':
                phi,allmat_phi = eigen_collect.multi_group(self,self.total,self.scatter,sources,phi_old,tol=1e-08,MAX_ITS=MAX_ITS)
                np.save('mydata/track_ae_{}/phi_{:<02}_count_{}'.format(problem,enrich,str(count).zfill(3)),allmat_phi)
            else:
                phi = eigen_collect.multi_group(self,self.total,self.scatter,sources,phi_old,tol=1e-08,MAX_ITS=MAX_ITS)
            
            if self.track == 'power':
                complete_phi = np.vstack((complete_phi,phi))

            keff = np.linalg.norm(phi)
            phi /= keff

            change = np.linalg.norm((phi-phi_old)/phi/(self.I))
            if LOUD:
                print('Change is',change,'Keff is',keff)
                print('===================================')
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1

            phi_old = phi.copy()
            sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old)
        if self.track == 'power':
            return phi,keff,complete_phi
        return phi,keff

class source:
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

    def transport(self,coder,problem='carbon',tol=1e-08,MAX_ITS=1500,normal=True):
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
        from discrete1.theProcess import func,ex_sources

        phi_old = func.initial_flux(problem)

        if self.track == 'source':
            allmat_phi = []
            allmat_smult = []; allmat_fmult = []

        smult = np.einsum('ijk,ik->ij',self.scatter,phi_old)
        fmult = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old)
        mult = smult + fmult
        external = ex_sources.source1(self.I,self.G)

        converged = 0
        count = 1
        if self.track == 'source':
            allmat_phi.append(phi_old)
            allmat_fmult.append(fmult)
            allmat_smult.append(smult)
        while not (converged):
            print('Source Iteration {}'.format(count))
            phi = np.zeros(phi_old.shape)  
            for g in range(self.G):
                phi[:,g] = source.one_group(self,self.total[:,g],mult[:,g],external,phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
            if self.track == 'source':
                allmat_phi.append(phi)
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            print('Change is',change,'\n===================================')
                
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 

            phi_old = phi.copy()

            smult = np.einsum('ijk,ik->ij',self.scatter,phi)
            fmult = np.einsum('ijk,ik->ij',self.chiNuFission,phi)
            if self.track == 'source':
                allmat_fmult.append(fmult)
                allmat_smult.append(smult)
            mult = smult + fmult
        if self.track == 'source':
            return phi,np.concatenate((allmat_phi)),np.concatenate((allmat_smult)),np.concatenate((allmat_fmult))
        return phi
 