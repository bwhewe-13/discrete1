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
        distance = [45,35,20]
        # distance = [45,5,25,5,20]
        delta = 0.1
        layers = [int(ii/delta) for ii in distance]
        if symm:
            splits = sn.layer_slice(layers,half=False)
        else:
            splits = sn.layer_slice(layers)
        # conc is uh3 enrich while 0 is depleted uranium
        enrichment = sn.enrich_list(sum(layers),[conc,0],[splits[1],splits[2]])
        return enrichment,sn.layer_slice_dict(layers,half=not symm)
    
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
        import ctypes
        clibrary = ctypes.cdll.LoadLibrary('./discrete1/cfunctions.so')
        sweep = clibrary.sweep
        converged = 0
        count = 1
        phi = np.zeros((self.I),dtype='float64')
        phi_old = guess.copy()
        half_total = 0.5*total.copy()
        external = external.astype('float64')
        while not(converged):
            phi = np.zeros((self.I))
            for n in range(self.N):
                weight = self.mu[n]*self.inv_delta
                top_mult = (weight-half_total).astype('float64')
                bottom_mult = (1/(weight+half_total)).astype('float64')
                if self.dtype == 'scatter' or self.dtype == 'both':
                    # djinn_1g should be length I'
                    temp_scat = djinn_1g.astype('float64')
                else:
                    temp_scat = (scatter * phi_old).astype('float64')
                # Set Pointers for C function
                phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)
                ext_ptr = ctypes.c_void_p(external.ctypes.data)
                top_ptr = ctypes.c_void_p(top_mult.ctypes.data)
                bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)
                # sweep(ctypes.c_void_p(phi.ctypes.data),ctypes.c_void_p(temp_scat.ctypes.data),ctypes.c_void_p(external.ctypes.data),ctypes.c_void_p(top_mult.ctypes.data),ctypes.c_void_p(bottom_mult.ctypes.data),ctypes.c_double(self.w[n]))
                sweep(phi_ptr,ts_ptr,ext_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]))
                # psi_bottom = 0 # vacuum on LHS
                # # Left to right
                # for ii in range(self.I):
                #     psi_top = (temp_scat[ii] + external[ii] + psi_bottom * top_mult[ii])*(bottom_mult[ii])
                #     phi[ii] = phi[ii] + (self.w[n] * func.diamond_diff(psi_top,psi_bottom))
                #     psi_bottom = psi_top
                # # Reflective right to left
                # for ii in range(self.I-1,-1,-1):
                #     psi_top = psi_bottom
                #     psi_bottom = (temp_scat[ii] + external[ii] + psi_top * top_mult[ii])*(bottom_mult[ii])
                #     phi[ii] = phi[ii] +  (self.w[n]* func.diamond_diff(psi_top,psi_bottom))
            
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi
        
    def multi_group(self,total,scatter,chiNuFission,model,tol=1e-08,MAX_ITS=100,initial=None):
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
        if self.track == 'scatter':
            allmat_sca = np.zeros((2,0,self.G+1))
        while not (converged):
            phi = np.zeros(phi_old.shape)
            if self.dtype == 'scatter' or self.dtype == 'both':
                if count == 1: # and initial is not None: #initialize 
                    # print('Initializing Scattering DJINN')
                    djinn_scatter_ns = eigen_djinn_symm.label_model(self,initial,model)
                    # print(djinn_scatter_ns.shape)
                    djinn_scatter = eigen_djinn_symm.scale_scatter(self,initial,djinn_scatter_ns)
                else:
                    djinn_scatter_ns = eigen_djinn_symm.label_model(self,phi_old,model)
                    djinn_scatter = eigen_djinn_symm.scale_scatter(self,phi_old,djinn_scatter_ns)
                for g in range(self.G):
                    phi[:,g] = eigen_djinn_symm.one_group(self,total[:,g],scatter[:,g,g],djinn_scatter[:,g],chiNuFission[:,g],phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
                if self.track == 'scatter':
                    temp_fission,temp_scatter = eigen_djinn_symm.tracking_data(self,phi,None)
                    allmat_sca = np.hstack((allmat_sca,temp_scatter))
            elif self.dtype == 'fission':
                for g in range(self.G):
                    if g == 0:
                        q_tilde = chiNuFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g)
                    else:
                        q_tilde = chiNuFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g) + func.update_q(scatter,phi,0,g,g)
                    phi[:,g] = eigen_djinn_symm.one_group(self,total[:,g],scatter[:,g,g],None,q_tilde,phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        if self.track == 'scatter':
            return phi, allmat_sca
        return phi
    
    def label_model(self,phi,model_):
        import numpy as np
        from discrete1.util import sn
        if np.sum(phi) == 0:
            return np.zeros((phi.shape))
            # return np.zeros((sn.cat(phi,self.splits['djinn']).shape))
        short_phi = sn.cat(phi,self.splits['djinn'])
        short_phi = phi.copy()
        if self.process == 'norm':
            short_phi /= np.linalg.norm(short_phi,axis=1)[:,None]
        if self.label:
            short_phi = np.hstack((self.enrich[:,None],short_phi))
        return model_.predict(short_phi) 

    def scale_scatter(self,phi,djinn_ns):
        import numpy as np
        from discrete1.util import sn
        if np.sum(phi) == 0:
            return np.zeros((phi.shape))
            # return np.zeros((sn.cat(phi,self.splits['djinn']).shape))
        # interest = sn.cat(phi,self.splits['djinn'])
        interest = phi.copy()
        scale = np.sum(interest*np.sum(self.scatter,axis=1),axis=1)/np.sum(djinn_ns,axis=1)
        return scale[:,None]*djinn_ns

    def scale_fission(self,phi,djinn_ns):
        import numpy as np
        from discrete1.util import sn
        if np.sum(phi) == 0:
            return np.zeros((phi.shape))
            # return np.zeros((sn.cat(phi,self.splits['djinn']).shape))
        if self.dtype == 'scatter':
            return np.einsum('ijk,ik->ij',self.chiNuFission,phi) 
        interest = sn.cat(phi,self.splits['djinn'])
        scale = np.sum(interest*np.sum(sn.cat(self.chiNuFission,self.splits['djinn']),axis=1),axis=1)/np.sum(djinn_ns,axis=1)
        regular = np.einsum('ijk,ik->ij',sn.cat(self.chiNuFission,self.splits['keep']),sn.cat(phi,self.splits['keep']))
        return np.vstack((regular,scale[:,None]*djinn_ns))
        
    def tracking_data(self,phi,sources=None):
        from discrete1.util import sn
        import numpy as np
        # labels = sn.cat(self.enrich,self.splits['djinn'])
        labels = self.enrich.copy()
        # short_phi = sn.cat(phi,self.splits['djinn'])
        short_phi = phi.copy()
        labeled_phi = np.hstack((labels[:,None],short_phi))
        if self.track == 'scatter':
            multiplier = np.einsum('ijk,ik->ij',self.scatter,phi)
            labeled_mult = np.hstack((labels[:,None],multiplier))
            return None,np.vstack((labeled_phi[None,:,:],labeled_mult[None,:,:]))
            # print(allmat_sca.shape)
        elif self.track == 'fission':
            short_fission = np.hstack((labels[:,None],sources))
            return np.vstack((labeled_phi[None,:,:],short_fission[None,:,:])),None
        multiplier = np.einsum('ijk,ik->ij',self.scatter,phi)
        labeled_mult = np.hstack((labels[:,None],multiplier))
        short_fission = np.hstack((labels[:,None],sources))
        return np.vstack((labeled_phi[None,:,:],short_fission[None,:,:])),np.vstack((labeled_phi[None,:,:],labeled_mult[None,:,:]))
        
    def transport(self,model_name,process,tol=1e-12,MAX_ITS=100,LOUD=True):
        """ EIGEN DJINN SYMM
        Arguments:
            model_name: File location of DJINN model
            process: should be 'norm' to normalize data
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a I x G array    """        
        import numpy as np
        from discrete1.util import sn
        self.process = process
        phi_old = np.random.rand(self.I,self.G)
        phi_old /= np.linalg.norm(phi_old)
        # Load DJINN model            
        model_scatter,model_fission = sn.djinn_load(model_name,self.dtype)
        # Label and predict the fission data
        dj_init = phi_old.copy()
        # dj_init = np.load('mydata/djinn_true_1d/phi2_15.npy')
        # phi_old = dj_init.copy() 
        if self.process == 'norm':
            dj_init /= np.linalg.norm(dj_init,axis=1)[:,None]
        if self.dtype == 'both' or self.dtype == 'fission':
            djinn_fission_ns = eigen_djinn_symm.label_model(self,dj_init,model_fission)
        else:
            djinn_fission_ns = 0
        # Scale DJINN fission or Get original sources      
        sources = eigen_djinn_symm.scale_fission(self,dj_init,djinn_fission_ns)
        converged = 0
        count = 1            
        while not (converged):             
            if LOUD:
                print('Outer Transport Iteration {}\n==================================='.format(count))
            if count == 1:
                initial = dj_init
            else:
                initial = phi_old.copy()
                # initial = np.load('mydata/djinn_true_1d/phi2_15.npy')
            if self.track == 'scatter':
                phi,temp_scatter = eigen_djinn_symm.multi_group(self,self.total,self.scatter,sources,model_scatter,tol=1e-08,MAX_ITS=MAX_ITS,initial=initial)
                enrich = str(np.amax(self.enrich)).split('.')[1]
                np.save('mydata/track_scatter_reg_plastic/enrich_{:<02}_count_{}'.format(enrich,str(count).zfill(3)),temp_scatter)
            else:    
                phi = eigen_djinn_symm.multi_group(self,self.total,self.scatter,sources,model_scatter,tol=1e-08,MAX_ITS=100,initial=initial)
            keff = np.linalg.norm(phi)
            phi /= keff
            change = np.linalg.norm((phi-phi_old)/phi/(self.I))
            if LOUD:
                print('Change is',change,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS)
            count += 1
            phi_old = phi.copy()
            # Update DJINN Models
            if self.dtype == 'both' or self.dtype == 'fission':
                djinn_fission_ns = eigen_djinn_symm.label_model(self,phi_old,model_fission)
            # Scale DJINN fission or Get original sources
            sources = eigen_djinn_symm.scale_fission(self,phi_old,djinn_fission_ns)
        # if self.track:
            # return phi,keff,allmat_fis,allmat_sca
        return phi,keff    
