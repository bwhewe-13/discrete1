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
    
class problem:        
    def variables(conc=None,ptype=None,distance=None,symm=False):
        from discrete1.util import chem,sn
        import numpy as np
        if ptype is None or ptype == 'ss440' or ptype == 'carbon_full' or ptype == 'carbon':
            distance = [45,35,20]
        elif ptype == 'multiplastic':
            distance = [10]*8; distance.append(20)
        elif ptype == 'mixed1':
            distance = [45,5,25,5,20]
            conc = [0.12,0.27]
        elif ptype == 'noplastic':
            distance = [35,20]
        elif ptype == 'ss440_flip':
            distance = [20,35,45]
        delta = 0.1
        if conc is None:
            conc = 0.2
        print('Concentration: ',conc)
        # Layer densities
        density_uh3 = 10.95; density_ch3 = 0.97
        if ptype == 'mixed1':
            uh3_density_low = chem.density_list('UH3',density_uh3,conc[0])
            uh3_density = chem.density_list('UH3',density_uh3,conc[1])
        else:
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
        if ptype == 'mixed1':
            uh3_scatter_low = uh3_density_low[0]*u235scatter + uh3_density_low[1]*u238scatter + uh3_density_low[2]*h1scatter
        hdpe_scatter = hdpe_density[0]*c12scatter + hdpe_density[1]*h1scatter
        uh3_238_scatter = uh3_238_density[0]*u238scatter + uh3_238_density[1]*h1scatter
    
        # Total Cross Section
        u235total = np.load('mydata/u235/vecTotal.npy')[eval(spec_temp)]
        u238total = np.load('mydata/u238/vecTotal.npy')[eval(spec_temp)]
        h1total = np.load('mydata/h1/vecTotal.npy')[eval(spec_temp)]
        c12total = np.load('mydata/cnat/vecTotal.npy')[eval(spec_temp)]
    
        uh3_total = uh3_density[0]*u235total + uh3_density[1]*u238total + uh3_density[2]*h1total
        if ptype == 'mixed1':
            uh3_total_low = uh3_density_low[0]*u235total + uh3_density_low[1]*u238total + uh3_density_low[2]*h1total
        hdpe_total = hdpe_density[0]*c12total + hdpe_density[1]*h1total
        uh3_238_total = uh3_238_density[0]*u238total + uh3_238_density[1]*h1total
    
        # Fission Cross Section
        u235fission = np.load('mydata/u235/nufission_0{}.npy'.format(spec_temp))[0]
        u238fission = np.load('mydata/u238/nufission_0{}.npy'.format(spec_temp))[0]
    
        uh3_fission = uh3_density[0]*u235fission + uh3_density[1]*u238fission
        if ptype == 'mixed1':
            uh3_fission_low = uh3_density_low[0]*u235fission + uh3_density_low[1]*u238fission
        uh3_238_fission = uh3_238_density[0]*u238fission
        hdpe_fission = np.zeros((dim,dim))
    
        # Cross section layers
        if ptype is None or ptype == 'blur' or ptype == 'carbon_full' or ptype == 'carbon':
            xs_scatter = [hdpe_scatter.T,uh3_scatter.T,uh3_238_scatter.T]
            xs_total = [hdpe_total,uh3_total,uh3_238_total]
            xs_fission = [hdpe_fission.T,uh3_fission.T,uh3_238_fission.T]
        elif ptype == 'mixed1':
            xs_scatter = [hdpe_scatter.T,uh3_scatter_low.T,uh3_scatter.T,uh3_scatter_low.T,uh3_238_scatter.T]
            xs_total = [hdpe_total,uh3_total_low,uh3_total,uh3_total_low,uh3_238_total]
            xs_fission = [hdpe_fission.T,uh3_fission_low.T,uh3_fission.T,uh3_fission_low.T,uh3_238_fission.T]
        elif ptype == 'multiplastic':
            xs_scatter = []; xs_total = []; xs_fission = []
            for ii in range(4):
                xs_scatter.append(hdpe_scatter.T); xs_scatter.append(uh3_scatter.T)
                xs_total.append(hdpe_total); xs_total.append(uh3_total)
                xs_fission.append(hdpe_fission.T); xs_fission.append(uh3_fission.T)
            xs_scatter.append(uh3_238_scatter.T)
            xs_total.append(uh3_238_total)
            xs_fission.append(uh3_238_fission.T)
        elif ptype == 'ss440':
            print('Using Stainless Steel')
            ss_total,ss_scatter = chem.xs_ss440(dim, spec_temp)
            xs_scatter = [ss_scatter.T,uh3_scatter.T,uh3_238_scatter.T]
            xs_total = [ss_total,uh3_total,uh3_238_total]
            xs_fission = [hdpe_fission.T,uh3_fission.T,uh3_238_fission.T]
        elif ptype == 'ss440_flip':
            print('Using FLIPPED Stainless Steel')
            ss_total,ss_scatter = chem.xs_ss440(dim, spec_temp)
            xs_scatter = [uh3_238_scatter.T,uh3_scatter.T,ss_scatter.T]
            xs_total = [uh3_238_total,uh3_total,ss_total]
            xs_fission = [uh3_238_fission.T,uh3_fission.T,hdpe_fission.T]
        elif ptype == 'noplastic':
            xs_scatter = [uh3_scatter.T,uh3_238_scatter.T]
            xs_total = [uh3_total,uh3_238_total]
            xs_fission = [uh3_fission.T,uh3_238_fission.T]
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
    
    def boundaries_aux(conc,ptype=None,distance=None,symm=False):
        import numpy as np
        from discrete1.util import sn
        if ptype == 'carbon' or ptype == 'ss440':
            distance = [45,35,20]
            ment = [conc,0]
            where = [1,2]
        elif ptype == 'multiplastic':
            distance = [10]*8; distance.append(20)
            ment = [conc]*4; ment.append(0)
            where = [1,3,5,7,8]
        elif ptype ==  'mixed1':
            distance = [45,5,25,5,20]
            ment = [0.12,0.27,0.12,0]
            where = [1,2,3,4]
        elif ptype == 'blur':
            distance = [47,33,20]
            ment = [conc,0]
            where = [1,2]
        elif ptype == 'noplastic':
            distance = [35,20]
            ment = [conc,0]
            where = [0,1]
        elif ptype == 'ss440_flip':
            distance = [20,35,45]
            ment = [0,conc]
            where = [0,1]
        elif ptype == 'carbon_full' or ptype == 'ss440_full':
            distance = [45,35,20]
            ment = [15.04,conc,0]
            where = [0,1,2]
        elif ptype == 'ss440_full':
            distance = [45,35,20]
            ment = [52.68,conc,0]
            where = [0,1,2]
        delta = 0.1
        layers = [int(ii/delta) for ii in distance]
        splits = np.array(sn.layer_slice(layers))
        # conc is uh3 enrich while 0 is depleted uranium
        enrichment = sn.enrich_list(sum(layers),ment,splits[where].tolist())
        return enrichment,sn.layer_slice_dict(layers,where)
    
    def boundaries(conc,ptype1=None,ptype2=None,distance=None,symm=False):
        """ ptype1 is fission model, ptype2 is scatter model """
        enrichment,splits = problem.boundaries_aux(conc,ptype1,distance,symm)
        fission_splits = {f'fission_{kk}': vv for kk, vv in splits.items()}
        enrichment,splits = problem.boundaries_aux(conc,ptype2,distance,symm)
        scatter_splits = {f'scatter_{kk}': vv for kk, vv in splits.items()}
        combo_splits = {**scatter_splits, **fission_splits}
        return enrichment,combo_splits
     
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
                # sweep(ctypes.c_void_p(phi.ctypes.data),ctypes.c_void_p(temp_scat.ctypes.data),ctypes.c_void_p(external.ctypes.data),ctypes.c_void_p(top_mult.ctypes.data),ctypes.c_void_p(bottom_mult.ctypes.data),ctypes.c_double(self.w[n]))
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
        if self.track == 'scatter':
            allmat_sca = np.zeros((2,0,self.G+1))
        while not (converged):
            phi = np.zeros(phi_old.shape)
            for g in range(self.G):
                if g == 0:
                    q_tilde = nuChiFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g)
                else:
                    q_tilde = nuChiFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g) + func.update_q(scatter,phi,0,g,g)
                phi[:,g] = eigen_symm.one_group(self,total[:,g],scatter[:,g,g],q_tilde,tol=tol,MAX_ITS=MAX_ITS,guess=phi_old[:,g])
            if self.track == "scatter":
                temp_fission,temp_scatter = eigen_symm.tracking_data(self,phi,np.empty((1000,87)))
                allmat_sca = np.hstack((allmat_sca,temp_scatter))
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        if self.track == 'scatter':
            return phi,allmat_sca
        return phi
            
    def tracking_data(self,phi,sources=None):
        from discrete1.util import sn
        import numpy as np
        labels = sn.cat(self.enrich,self.splits)
        # print(labels.shape)
        short_phi = sn.cat(phi,self.splits)
        labeled_phi = np.hstack((labels[:,None],short_phi))
        # if self.track == 'scatter':
        #     multiplier = sn.cat(np.einsum('ijk,ik->ij',self.scatter,phi),self.splits)
        #     labeled_mult = np.hstack((labels[:,None],multiplier))
        #     return None,np.vstack((labeled_phi[None,:,:],labeled_mult[None,:,:]))
        #     # print(allmat_sca.shape)
        # elif self.track == 'fission':
        #     short_fission = np.hstack((labels[:,None],sn.cat(sources,self.splits)))
        #     return np.vstack((labeled_phi[None,:,:],short_fission[None,:,:])),None
        multiplier = sn.cat(np.einsum('ijk,ik->ij',self.scatter,phi),self.splits)
        labeled_mult = np.hstack((labels[:,None],multiplier))
        # print(sources.shape,self.splits)
        # print(labels.shape)
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
        if self.track == 'scatter':
            temp_fission2,temp_scatter2 = eigen_symm.tracking_data(self,phi_old,sources)
            enrich = str(np.amax(self.enrich)).split('.')[1]
            np.save('mydata/track_stainless/enrich_{:<02}_count_000'.format(enrich),temp_scatter2)
        while not (converged):
            if self.track:
                temp_fission,temp_scatter = eigen_symm.tracking_data(self,phi_old,sources)
                # print(temp_fission.shape,temp_scatter.shape)
                allmat_sca = np.hstack((allmat_sca,temp_scatter))
                allmat_fis = np.hstack((allmat_fis,temp_fission))
            print('Outer Transport Iteration {}\n==================================='.format(count))
            if self.track == 'scatter':
                phi,temp_scatter2 = eigen_symm.multi_group(self,self.total,self.scatter,sources,tol=1e-08,MAX_ITS=MAX_ITS)
                enrich = str(np.amax(self.enrich)).split('.')[1]
                np.save('mydata/track_stainless/enrich_{:<02}_count_{}'.format(enrich,str(count).zfill(3)),temp_scatter2)
                # allmat_sca = np.hstack((allmat_sca,temp_scatter))
            else:
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
                    
    def one_group(self,total,scatter,djinn_1g,model,group,totalPhi,external,guess,tol=1e-08,MAX_ITS=100):
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
                if (self.dtype == 'scatter' or self.dtype == 'both'):
                    # djinn_scatter_ns = eigen_djinn_symm.label_model(self,phi_old,model)
                    # djinn_scatter = eigen_djinn_symm.scale_scatter(self,phi_old,djinn_scatter_ns)
                    # temp_scat = np.concatenate([sn.cat(scatter*phi_old,self.splits['keep']),djinn_scatter]).astype('float64')
                    # djinn_1g should be length I'
                    temp_scat = sn.pops_robust('scatter',(self.I,),sn.cat(scatter*phi_old,self.splits['scatter_keep']),djinn_1g,self.splits).astype('float64')
                    # temp_scat = np.concatenate([sn.cat(scatter*phi_old,self.splits['keep']),djinn_1g]).astype('float64')
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
            # if self.dtype == 'scatter' and count > 30:
                # # print('Update {} {}'.format(count,group))
                # djinn_1g = eigen_djinn_symm.update_phi(self,totalPhi,phi,group,model)                
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi
        
    def update_phi(self,CompletePhi,phi,g,model):
        CompletePhi[:,g] = phi.copy()
        djinn_scatter_ns = eigen_djinn_symm.label_model(self,CompletePhi,model)
        djinn_scatter = eigen_djinn_symm.scale_scatter(self,CompletePhi,djinn_scatter_ns)
        return djinn_scatter[:,g]
    
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
                if (np.sum(phi_old) == 0):
                    djinn_scatter = np.zeros(phi_old.shape)
                else:
                    djinn_scatter_ns = eigen_djinn_symm.label_model(self,'scatter',phi_old,model)
                    djinn_scatter = eigen_djinn_symm.scale_scatter(self,phi_old,djinn_scatter_ns)
                # if count == 1: # and initial is not None: #initialize 
                #     # print('Initializing Scattering DJINN')
                #     djinn_scatter_ns = eigen_djinn_symm.label_model(self,initial,model)
                #     djinn_scatter = eigen_djinn_symm.scale_scatter(self,initial,djinn_scatter_ns)
                #     # complete_phi = initial.copy()
                #     # print('Got Here')
                # else:
                #     djinn_scatter_ns = eigen_djinn_symm.label_model(self,phi_old,model)
                #     djinn_scatter = eigen_djinn_symm.scale_scatter(self,phi_old,djinn_scatter_ns)
                    # complete_phi = phi_old.copy()                
                for g in range(self.G):
                    phi[:,g] = eigen_djinn_symm.one_group(self,total[:,g],scatter[:,g,g],djinn_scatter[:,g],model,g,None,chiNuFission[:,g],phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
                    # complete_phi[:,g] = phi[:,g].copy()
                if self.track == 'scatter':
                    temp_fission,temp_scatter = eigen_djinn_symm.tracking_data(self,phi,None)
                    allmat_sca = np.hstack((allmat_sca,temp_scatter))
            elif self.dtype == 'fission':
                for g in range(self.G):
                    if g == 0:
                        q_tilde = chiNuFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g)
                    else:
                        q_tilde = chiNuFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g) + func.update_q(scatter,phi,0,g,g)
                    phi[:,g] = eigen_djinn_symm.one_group(self,total[:,g],scatter[:,g,g],None,None,None,None,q_tilde,phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        if self.track == 'scatter':
            return phi, allmat_sca
        return phi
    
    def label_model(self,xs,phi,model_):
        import numpy as np
        from discrete1.util import sn
        if np.sum(phi) == 0:
            return np.zeros((sn.cat(phi,self.splits['{}_djinn'.format(xs)]).shape))
        short_phi = sn.cat(phi,self.splits['{}_djinn'.format(xs)])
        # if self.process == 'norm':
        #     short_phi /= np.linalg.norm(short_phi,axis=1)[:,None]
        if self.label:
            short_phi = np.hstack((sn.cat(self.enrich,self.splits['{}_djinn'.format(xs)])[:,None],short_phi))
        return model_.predict(short_phi) 

    def scale_scatter(self,phi,djinn_ns):
        import numpy as np
        from discrete1.util import sn
        if np.sum(phi) == 0:
            return np.zeros((sn.cat(phi,self.splits['scatter_djinn']).shape))
        interest = sn.cat(phi,self.splits['scatter_djinn'])
        scale = np.sum(interest*np.sum(sn.cat(self.scatter,self.splits['scatter_djinn']),axis=1),axis=1)/np.sum(djinn_ns,axis=1)
        return scale[:,None]*djinn_ns

    def scale_fission(self,phi,djinn_ns):
        import numpy as np
        from discrete1.util import sn
        if np.sum(phi) == 0:
            return np.zeros((sn.cat(phi,self.splits['fission_djinn']).shape))
        if self.dtype == 'scatter':
            return np.einsum('ijk,ik->ij',self.chiNuFission,phi) 
        interest = sn.cat(phi,self.splits['fission_djinn'])
        scale = np.sum(interest*np.sum(sn.cat(self.chiNuFission,self.splits['fission_djinn']),axis=1),axis=1)/np.sum(djinn_ns,axis=1)
        # All of the sigma*phi terms not calculated by DJINN
        regular = np.einsum('ijk,ik->ij',sn.cat(self.chiNuFission,self.splits['fission_keep']),sn.cat(phi,self.splits['fission_keep']))
        return sn.pops_robust('fission',phi.shape,regular,scale[:,None]*djinn_ns,self.splits)
        
    def tracking_data(self,phi,sources=None):
        from discrete1.util import sn
        import numpy as np
        labels = sn.cat(self.enrich,self.splits['djinn'])
        short_phi = sn.cat(phi,self.splits['djinn'])
        labeled_phi = np.hstack((labels[:,None],short_phi))
        if self.track == 'scatter':
            multiplier = sn.cat(np.einsum('ijk,ik->ij',self.scatter,phi),self.splits['djinn'])
            labeled_mult = np.hstack((labels[:,None],multiplier))
            return None,np.vstack((labeled_phi[None,:,:],labeled_mult[None,:,:]))
            # print(allmat_sca.shape)
        elif self.track == 'fission':
            short_fission = np.hstack((labels[:,None],sn.cat(sources,self.splits['djinn'])))
            return np.vstack((labeled_phi[None,:,:],short_fission[None,:,:])),None
        multiplier = sn.cat(np.einsum('ijk,ik->ij',self.scatter,phi),self.splits['djinn'])
        labeled_mult = np.hstack((labels[:,None],multiplier))
        short_fission = np.hstack((labels[:,None],sn.cat(sources,self.splits['djinn'])))
        return np.vstack((labeled_phi[None,:,:],short_fission[None,:,:])),np.vstack((labeled_phi[None,:,:],labeled_mult[None,:,:]))
        
    def transport(self,model_name,process,ptype=None,tol=1e-12,MAX_ITS=100,LOUD=True):
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
        # phi_old = np.random.rand(self.I,self.G)
        # phi_old /= np.linalg.norm(phi_old)
        # Load DJINN model            
        model_scatter,model_fission = sn.djinn_load(model_name,self.dtype)
        # Label and predict the fission data
        if ptype is None or ptype == 'carbon_full' or ptype == 'carbon':
            dj_init = np.load('discrete1/data/phi_orig_15.npy')
        elif ptype == 'ss440':
            dj_init = np.load('discrete1/data/phi_ss_15.npy')
        elif ptype == 'ss440_flip':
            dj_init = np.load('discrete1/data/phi_ss_flip_15.npy')
        elif ptype == 'multiplastic':
            dj_init = np.load('discrete1/data/phi_mp_15.npy')
        phi_old = dj_init.copy() 
        if self.process == 'norm':
            dj_init /= np.linalg.norm(dj_init,axis=1)[:,None]
        if self.dtype == 'both' or self.dtype == 'fission':
            djinn_fission_ns = eigen_djinn_symm.label_model(self,'fission',dj_init,model_fission)
        else:
            djinn_fission_ns = 0
        # Scale DJINN fission or Get original sources
        # if self.track:
        #     allmat_sca = np.zeros((2,0,self.G+1))
        #     allmat_fis = np.zeros((2,0,self.G+1))        
        sources = eigen_djinn_symm.scale_fission(self,dj_init,djinn_fission_ns)
        converged = 0
        count = 1         
        while not (converged):
            # if self.track:
                # temp_fission,temp_scatter = eigen_djinn_symm.tracking_data(self,phi_old,sources)
                # allmat_sca = np.hstack((allmat_sca,temp_scatter))
                # allmat_fis = np.hstack((allmat_fis,temp_fission))                
            if LOUD:
                print('Outer Transport Iteration {}\n==================================='.format(count))
            # if count == 1:
            #     initial = dj_init
            # else:
            #     initial = phi_old.copy()
                # initial = np.load('mydata/djinn_true_1d/phi2_15.npy')
            if self.track == 'scatter':
                phi,temp_scatter = eigen_djinn_symm.multi_group(self,self.total,self.scatter,sources,model_scatter,tol=1e-08,MAX_ITS=MAX_ITS,initial=phi_old)
                enrich = str(np.amax(self.enrich)).split('.')[1]
                np.save('mydata/track_scatter_reg_std/enrich_{:<02}_count_{}'.format(enrich,str(count).zfill(3)),temp_scatter)
            else:    
                phi = eigen_djinn_symm.multi_group(self,self.total,self.scatter,sources,model_scatter,tol=1e-08,MAX_ITS=100,initial=phi_old)
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
                djinn_fission_ns = eigen_djinn_symm.label_model(self,'fission',phi_old,model_fission)
            # Scale DJINN fission or Get original sources
            sources = eigen_djinn_symm.scale_fission(self,phi_old,djinn_fission_ns)
        # if self.track:
            # return phi,keff,allmat_fis,allmat_sca
        return phi,keff    
