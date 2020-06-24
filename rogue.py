class func:
    def total_add(tot,mu,delta,side):
        if side == 'right' and mu > 0: #forward sweep
            return mu/delta-(0.5*tot)
        elif side == 'right' and mu < 0: #backward sweep
            return -mu/delta-(0.5*tot)
        elif side == 'left' and mu > 0: #forward sweep
            return mu/delta+(0.5*tot)
        return -mu/delta+(0.5*tot) #backward sweep
    
    def diamond_diff(top,bottom):
        return 0.5*(top + bottom)

    def update_q(scatter,phi,start,stop,g):
        import numpy as np
        return np.sum(scatter[:,g,start:stop]*phi[:,start:stop],axis=1)
    
class eigen_djinn:        
    def variables(conc=None,distance=None,symm=False):
        from discrete1.util import chem,sn
        import numpy as np
        
        distance = [45,35,20]
        delta = 0.1
        if conc is None:
            conc = 0.2
        print('Concentration: ',conc)
        # Layer densities
        density_uh3 = 10.95; density_ch3 = 0.97
        uh3_density = chem.density_list('UH3',density_uh3,conc)
        hdpe_density = chem.density_list('CH3',density_ch3)
        uh3_238_density = chem.density_list('U^238H3',density_uh3)
    
        # Loading Cross Section Data
        dim = 87; spec_temp = '00'
        # Scattering Cross Section
        u235scatter = np.load('mydata/u235/scatter_0{}.npy'.format(spec_temp))[0]
        u238scatter = np.load('mydata/u238/scatter_0{}.npy'.format(spec_temp))[0]
        h1scatter = np.load('mydata/h1/scatter_0{}.npy'.format(spec_temp))[0]
        c12scatter = np.load('mydata/cnat/scatter_0{}.npy'.format(spec_temp))[0]
    
        uh3_scatter = uh3_density[0]*u235scatter + uh3_density[1]*u238scatter + uh3_density[2]*h1scatter
        hdpe_scatter = hdpe_density[0]*c12scatter + hdpe_density[1]*h1scatter
        uh3_238_scatter = uh3_238_density[0]*u238scatter + uh3_238_density[1]*h1scatter
    
        # Total Cross Section
        u235total = np.load('mydata/u235/vecTotal.npy')[eval(spec_temp)]
        u238total = np.load('mydata/u238/vecTotal.npy')[eval(spec_temp)]
        h1total = np.load('mydata/h1/vecTotal.npy')[eval(spec_temp)]
        c12total = np.load('mydata/cnat/vecTotal.npy')[eval(spec_temp)]
    
        uh3_total = uh3_density[0]*u235total + uh3_density[1]*u238total + uh3_density[2]*h1total
        hdpe_total = hdpe_density[0]*c12total + hdpe_density[1]*h1total
        uh3_238_total = uh3_238_density[0]*u238total + uh3_238_density[1]*h1total
    
        # Fission Cross Section
        u235fission = np.load('mydata/u235/nufission_0{}.npy'.format(spec_temp))[0]
        u238fission = np.load('mydata/u238/nufission_0{}.npy'.format(spec_temp))[0]
    
        uh3_fission = uh3_density[0]*u235fission + uh3_density[1]*u238fission
        uh3_238_fission = uh3_238_density[0]*u238fission
        hdpe_fission = np.zeros((dim,dim))
    
        # Cross section layers
        xs_scatter = [hdpe_scatter.T,uh3_scatter.T,uh3_238_scatter.T,uh3_scatter.T,hdpe_scatter.T]
        xs_total = [hdpe_total,uh3_total,uh3_238_total,uh3_total,hdpe_total]
        xs_fission = [hdpe_fission.T,uh3_fission.T,uh3_238_fission.T,uh3_fission.T,hdpe_fission.T]
    
        # Setting up eigenvalue equation
        N = 8; L = 0; R = sum(distance); G = dim
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w)
        if symm:
            mu = mu[int(N*0.5):]
            w = w[int(N*0.5):]
            N = int(N*0.5) 
        layers = [int(ii/delta) for ii in distance]
        I = int(sum(layers))
    
        scatter_ = sn.mixed_propagate(xs_scatter,layers,G=dim,L=L,dtype='scatter')
        fission_ = sn.mixed_propagate(xs_fission,layers,G=dim,dtype='fission2')
        total_ = sn.mixed_propagate(xs_total,layers,G=dim)
        
        return G,N,mu,w,total_,scatter_[:,0],fission_,L,R,I
    
    def boundaries(conc,distance=None,symm=False):
        # import numpy as np
        from discrete1.util import sn
        #distance = [50,35,40,35,50]
        #delta = 0.2
        # if distance is None:
        #     # distance = [200,75,150,75,200]
        #     distance = [45,35,40,35,45]
        #     if symm:
        #         # distance = [200,75,75]
        distance = [45,35,20]
        delta = 0.1
        layers = [int(ii/delta) for ii in distance]
        if symm:
            splits = sn.layer_slice(layers,half=False)
        else:
            splits = sn.layer_slice(layers)
        # conc is uh3 enrich while 0 is depleted uranium
        enrichment = sn.enrich_list(sum(layers),[conc,0],[splits[1],splits[2]])
        return enrichment,sn.layer_slice_dict(layers,half=not symm)
        
class eigen_symm:
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
        converged = 0
        count = 1        
        phi = np.zeros((self.I))
        phi_old = guess.copy()
        half_total = 0.5*total.copy()
        while not(converged):
            phi *= 0
            for n in range(self.N):
                temp_scat = scatter*phi_old
                weight = self.mu[n]*self.inv_delta
                psi_bottom = 0 # vacuum on LHS
                # Left to right
                for ii in range(self.I):
                    psi_top = (temp_scat[ii] + external[ii] + psi_bottom * (weight-half_total[ii]))/(weight+half_total[ii])
                    phi[ii] = phi[ii] + (self.w[n] * func.diamond_diff(psi_top,psi_bottom))
                    psi_bottom = psi_top
                # Reflective right to left
                for ii in range(self.I-1,-1,-1):
                    psi_top = psi_bottom
                    psi_bottom = (temp_scat[ii] + external[ii] + psi_top * (weight-half_total[ii]))/(weight+half_total[ii])
                    phi[ii] = phi[ii] +  (self.w[n] * func.diamond_diff(psi_top,psi_bottom))
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
        phi_old = np.zeros((self.I,self.G))
        converged = 0
        count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)
            for g in range(self.G):
                if g == 0:
                    q_tilde = nuChiFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g)
                else:
                    q_tilde = nuChiFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g) + func.update_q(scatter,phi,0,g,g)
                phi[:,g] = eigen_symm.one_group(self,total[:,g],scatter[:,g,g],q_tilde,tol=tol,MAX_ITS=MAX_ITS,guess=phi_old[:,g])
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi
            
    def tracking_data(self,phi,sources=None):
        from discrete1.util import sn
        import numpy as np
        labels = sn.cat(self.enrich,self.splits)
        short_phi = sn.cat(phi,self.splits)
        labeled_phi = np.hstack((labels[:,None],short_phi))
        if self.track == 'scatter':
            multiplier = sn.cat(np.einsum('ijk,ik->ij',self.scatter,phi),self.splits)
            labeled_mult = np.hstack((labels[:,None],multiplier))
            return None,np.vstack((labeled_phi[None,:,:],labeled_mult[None,:,:]))
            # print(allmat_sca.shape)
        elif self.track == 'fission':
            short_fission = np.hstack((labels[:,None],sn.cat(sources,self.splits)))
            return np.vstack((labeled_phi[None,:,:],short_fission[None,:,:])),None
        multiplier = sn.cat(np.einsum('ijk,ik->ij',self.scatter,phi),self.splits)
        labeled_mult = np.hstack((labels[:,None],multiplier))
        short_fission = np.hstack((labels[:,None],sn.cat(sources,self.splits)))
        return np.vstack((labeled_phi[None,:,:],short_fission[None,:,:])),np.vstack((labeled_phi[None,:,:],labeled_mult[None,:,:]))
            

    def transport(self,tol=1e-12,MAX_ITS=100,LOUD=True):
        """ Arguments:
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a I x G array    """        
        import numpy as np
        # from discrete1.util import sn
        phi_old = np.random.rand(self.I,self.G)
        k_old = np.linalg.norm(phi_old)
        track_k = [k_old]
        phi_old /= np.linalg.norm(phi_old)
        converged = 0
        count = 1
        if self.track:
            allmat_sca = np.zeros((2,0,self.G+1))
            allmat_fis = np.zeros((2,0,self.G+1))
        sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old) 
        while not (converged):
            if self.track:
                temp_fission,temp_scatter = eigen_symm.tracking_data(self,phi_old,sources)
                allmat_sca = np.hstack((allmat_sca,temp_scatter))
                allmat_fis = np.hstack((allmat_fis,temp_fission))
            print('Outer Transport Iteration {}\n==================================='.format(count))
            phi = eigen_symm.multi_group(self,self.total,self.scatter,sources,tol=1e-08,MAX_ITS=MAX_ITS)
            keff = np.linalg.norm(phi)
            phi /= keff
            kchange = abs(keff-k_old)
            track_k.append(keff)
            change = np.linalg.norm((phi-phi_old)/phi/(self.I))
            if LOUD:
                print('Change is',change,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS) or (kchange < tol)
            count += 1
            phi_old = phi.copy()
            k_old = keff
            sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old) 
        if self.track:
            return phi,track_k,allmat_fis,allmat_sca
        return phi,keff
    
class eigen_djinn_symm:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L,R,I,dtype='scatter',enrich=None,splits=None,track=None,label=None):
        """ splits are in a dictionary as compared to eigen where splits are a list
        dictionary items are 'djinn' and 'keep' """
        self.G = G
        self.N = N
        self.mu = mu
        self.w = w
        self.total = total
        self.scatter = scatter
        self.chiNuFission = chiNuFission
        self.L = L
        self.I = I
        self.inv_delta = float(I)/R
        self.dtype = dtype
        self.enrich = enrich
        self.splits = splits
        self.track = track
        self.label = label
        
    def check_scale(self,phi_old,djinn_scatter_ns,djinn_fission_ns):
        import numpy as np
        from discrete1.util import sn
        # Used for scaling
        normed = sn.cat(phi_old[:,0],self.splits['djinn'])
        if np.sum(phi_old) == 0:# or self.dtype == 'fission':
            lens = sn.length(self.splits['djinn'])
            return np.zeros((lens,self.G))
        elif self.dtype == 'scatter':# or self.dtype == 'both':
            return np.einsum('ijk,ik->ij',self.chiNuFission,phi_old) 
        elif self.dtype == 'fission' or self.dtype == 'both':
            scale = np.sum(normed*np.sum(sn.cat(self.chiNuFission,self.splits['djinn']),axis=1),axis=1)/np.sum(djinn_fission_ns,axis=1)
            non_scale = sn.cat(np.ones((self.I,self.G)),self.splits['keep'])
            sources = np.concatenate((non_scale,np.expand_dims(scale,axis=1)*djinn_fission_ns)).reshape(self.I,self.G)

        return sources
    
    def multi_scale(self,phi_old,model,scatter,G,I,L):
        import numpy as np
        from discrete1.util import sn
        if(np.sum(phi_old) == 0):
            lens = sn.length(self.splits['djinn'])
            dj_pred = np.zeros((lens,G))
            # dj_pred = np.zeros((I,L+1,G))
        else:
            normed = sn.cat(phi_old[:,0],self.splits['djinn'])
            if self.label is not None:
                # Not including L+1 dimension
                dj_pred_ns = model.predict(np.concatenate((np.expand_dims(sn.cat(self.enrich,self.splits['djinn']),axis=1),normed),axis=1)) 
            else:
                dj_pred_ns = model.predict(normed)
            # Should be I'x1 dimensions    
            scale = np.sum(normed*np.sum(sn.cat(scatter[:,0],self.splits['djinn']),axis=1),axis=1)/np.sum(dj_pred_ns,axis=1)
            # Other I dimensions
            # non_scale = sn.cat(np.ones((I,G)),self.splits['keep'])
            # dj_pred = np.concatenate((non_scale,np.expand_dims(scale,axis=1)*dj_pred_ns)).reshape(I,L+1,G) #I dimensions
            dj_pred = np.expand_dims(scale,axis=1)*dj_pred_ns; 
            # print(scale)
        return dj_pred
        
    def one_group(self,total,scatter,djinn_1g,external,guess,tol=1e-08,MAX_ITS=100):
        """ Arguments:
            total: I x 1 vector of the total cross section for each spatial cell
            scatter: I x 1 array for the scattering of the spatial cell by moment
            external: I x N array for the external sources
            guess: Initial guess of the scalar flux for a specific energy group (I x L+1)
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
        Returns:
            phi: a I x L+1 array   """
        import numpy as np        
        from discrete1.util import sn
        converged = 0
        count = 1
        phi = np.zeros((self.I))
        phi_old = guess.copy()
        half_total = 0.5*total.copy()
        while not(converged):
            phi *= 0
            for n in range(self.N):
                weight = self.mu[n]*self.inv_delta
                top_mult = weight-half_total
                bottom_mult = 1/(weight+half_total)
                if self.dtype == 'scatter' or self.dtype == 'both':
                    # djinn_1g should be length I'
                    temp_scat = np.concatenate([sn.cat(scatter*phi_old,self.splits['keep']),djinn_1g])
                else:
                    temp_scat = scatter * phi_old
                psi_bottom = 0 # vacuum on LHS
                # Left to right
                for ii in range(self.I):
                    psi_top = (temp_scat[ii] + external[ii] + psi_bottom * top_mult[ii])*(bottom_mult[ii])
                    phi[ii] = phi[ii] + (self.w[n] * func.diamond_diff(psi_top,psi_bottom))
                    psi_bottom = psi_top
                # Reflective right to left
                for ii in range(self.I-1,-1,-1):
                    psi_top = psi_bottom
                    psi_bottom = (temp_scat[ii] + external[ii] + psi_top * top_mult[ii])*(bottom_mult[ii])
                    phi[ii,:] = phi[ii] +  (self.w[n]* func.diamond_diff(psi_top,psi_bottom))
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi
    
    def multi_group(self,total,scatter,chiNuFission,model,tol=1e-08,MAX_ITS=100):
        # self,self.G,self.N,self.mu,self.w,self.total,self.scatter,self.L,sources,model_scatter,djinn_scatter,self.I,self.delta,
        """ Arguments:
            total: I x G vector of the total cross section for each spatial cell and energy level
            scatter: I x G array for the scattering of the spatial cell by moment and energy
            nuChiFission: 
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
        Returns:
            phi: a I x G array  """
        import numpy as np
        # from discrete1.util import sn
        phi_old = np.zeros((self.I,self.G))
        converged = 0
        count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)
            if self.dtype == 'scatter' or self.dtype == 'both':
                # Not going to be of size I'
                dj_pred = eigen_djinn_symm.multi_scale(self,phi_old,model,scatter,self.G,self.I)
                for g in range(self.G):
                    phi[:,g] = eigen_djinn_symm.one_group(self,total[:,g],scatter[:,g,g],dj_pred[:,g],chiNuFission[:,g],phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
            elif self.dtype == 'fission':
                for g in range(self.G):
                    if g == 0:
                        q_tilde = chiNuFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g)
                    else:
                        q_tilde = chiNuFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g) + func.update_q(scatter,phi,0,g,g)
                    phi[:,g] = eigen_djinn_symm.one_group(self,total[:,g],scatter[:,g,g],None,q_tilde,tol=tol,MAX_ITS=MAX_ITS,guess=phi_old[:,g])
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi

    def transport(self,model_name,tol=1e-12,MAX_ITS=100,LOUD=True):
        """ EIGEN DJINN SYMM
        Arguments:
            G: number of energy groups
            N: number of angles
            mu: vector of angles between -1 and 1
            w: vector of weights for the angles
            total: G x 1 vector of the total cross section for each energy level
            scatter: G x G array for the scattering of the spatial cell by energy
            chiNuFission: G x G array of chi x nu x sigma_f
            L: Number of moments
            I: Number of spatial cells
            delta: width of one spatial cell
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a I x L+1 x G array    """        
        import numpy as np
        # from djinn import djinn
        from discrete1.util import sn
        phi_old = np.random.rand(self.I,self.L+1,self.G)
        phi_old /= np.linalg.norm(phi_old)
        # Load DJINN model            
        model_scatter,model_fission = sn.djinn_load(model_name,self.dtype)
        # Check if enriched
        if self.label is None:
            djinn_scatter_ns,djinn_fission_ns = sn.enriche(phi_old,model_scatter,model_fission,self.dtype,self.splits['djinn'])
        else:
            djinn_scatter_ns,djinn_fission_ns = sn.enriche(phi_old,model_scatter,model_fission,self.dtype,self.splits['djinn'],enrich=self.enrich)
        # Retrieve only the phi_old from DJINN part
        sources = eigen_djinn_symm.check_scale(self,phi_old,djinn_scatter_ns,djinn_fission_ns) 
        if self.track:
            allmat_fis = np.zeros((0,3,self.G))
            allmat_sca = np.zeros((0,3,self.G))
        converged = 0
        count = 1            
        while not (converged):
            if self.track == 'scatter' or self.track == 'both':
                enrichment = np.expand_dims(np.tile(np.expand_dims(sn.cat(self.enrich,self.splits['djinn']),axis=1),(1,self.G)),axis=1)
                multiplier = np.expand_dims(np.einsum('ijk,ik->ij',sn.cat(self.scatter,self.splits['djinn'])[:,0],sn.cat(phi_old,self.splits['djinn'])[:,0]),axis=1)
                track_temp = np.hstack((enrichment,sn.cat(phi_old,self.splits['djinn']),multiplier))
                allmat_sca = np.vstack((allmat_sca,track_temp))
                # print(allmat_sca.shape)
            if self.track == 'fission' or self.track == 'both':
                enrichment = np.expand_dims(np.tile(np.expand_dims(sn.cat(self.enrich,self.splits['djinn']),axis=1),(1,self.G)),axis=1)
                multiplier = np.expand_dims(np.einsum('ijk,ik->ij',sn.cat(self.chiNuFission,self.splits['djinn']),sn.cat(phi_old,self.splits['djinn'])[:,0]),axis=1)
                track_temp = np.hstack((enrichment,sn.cat(phi_old,self.splits['djinn']),multiplier))
                allmat_fis = np.vstack((allmat_fis,track_temp))
            if LOUD:
                print('Outer Transport Iteration {}\n==================================='.format(count))
            phi = eigen_djinn_symm.multi_group(self,self.G,self.N,self.mu,self.w,self.total,self.scatter,self.L,sources,model_scatter,self.I,self.delta,tol=1e-08,MAX_ITS=MAX_ITS,LOUD=False)
            keff = np.linalg.norm(phi)
            phi /= np.linalg.norm(phi)
            change = np.linalg.norm((phi-phi_old)/phi/(self.I*(self.L+1)))
            if LOUD:
                print('Change is',change,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS)
            count += 1
            phi_old = phi.copy()
            # Update DJINN Models
            if self.label is None:
                djinn_scatter_ns,djinn_fission_ns = sn.enriche(phi_old,model_scatter,model_fission,self.dtype,self.splits['djinn'])
            else:
                djinn_scatter_ns,djinn_fission_ns = sn.enriche(phi_old,model_scatter,model_fission,self.dtype,self.splits['djinn'],enrich=self.enrich)
            sources = eigen_djinn_symm.check_scale(self,phi_old,djinn_scatter_ns,djinn_fission_ns)
        if self.track:
            return phi,keff,allmat_fis,allmat_sca
        return phi,keff    
