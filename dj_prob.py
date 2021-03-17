# class tools:
#     def djinn_load(model_name,dtype):
#         from djinn import djinn
#         # from dj2.djinn import djinn
#         if dtype == 'both':
#             model_scatter = djinn.load(model_name=model_name[0])
#             model_fission = djinn.load(model_name=model_name[1])
#         elif dtype == 'scatter':
#             model_scatter = djinn.load(model_name=model_name)
#             model_fission = None
#         elif dtype == 'fission':
#             model_scatter = None
#             model_fission = djinn.load(model_name=model_name)
#         return model_scatter,model_fission
     

class eigen_djinn:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L,R,I,enrich=None,splits=None,track=None,label=None):
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
        self.enrich = enrich
        self.splits = splits
        self.track = track
        self.label = label
        
        self.multDJ = 'scatter'
        self.double = False
    
    def label_model(self,xs,flux,model_,material='djinn'):
        import numpy as np
        from discrete1.util import sn
        # Take the phi of only the DJINN part
        short_phi = sn.cat(flux,self.splits['{}_{}'.format(xs,material)])
        # Check for labeling
        if self.label:
            short_phi = np.hstack((sn.cat(self.enrich,self.splits['{}_{}'.format(xs,material)])[:,None],short_phi))
        # Predict DJINN
        djinn_ns = model_.predict(short_phi)
        return djinn_ns 

    def scale_scatter(self,phi,djinn_ns,material='djinn'):
        import numpy as np
        from discrete1.util import sn
        # Calculate phi for DJINN part
        short_phi = sn.cat(phi,self.splits['scatter_{}'.format(material)])
        # Calculate scaling factor
        scale = np.sum(short_phi*np.sum(sn.cat(self.scatter,self.splits['scatter_{}'.format(material)]),axis=1),axis=1)/np.sum(djinn_ns,axis=1)
        # Scale DJINN
        djinn_s = scale[:,None]*djinn_ns
        # All of the sigma*phi terms not calculated by DJINN
        if not self.double:
            regular = np.einsum('ijk,ik->ij',sn.cat(self.scatter,self.splits['scatter_keep']),sn.cat(phi,self.splits['scatter_keep']))
            return sn.pops_robust('scatter',phi.shape,regular,djinn_s,self.splits)
        return djinn_s
        # return scale[:,None]*djinn_ns

    def create_smult(self,flux):
        import numpy as np
        from discrete1.util import sn
        from discrete1.setup_ke import func

        model_name = 'scatter_1d/cscat/djinn_reg/model_001003'
        self.model_scatter,self.model_fission = func.djinn_load(model_name,self.multDJ)

        # Check for Zeros
        if (np.sum(flux) == 0):
            return np.zeros(flux.shape)
        # Predict DJINN
        djinn_scatter_ns = eigen_djinn.label_model(self,'scatter',flux,self.model_scatter)
        djinn_scatter_s = eigen_djinn.scale_scatter(self,flux,djinn_scatter_ns)
        if self.double:
            # Do the same for reflective material
            djinn_refl_ns = eigen_djinn.label_model(self,'scatter',flux,self.refl_scatter,'keep')
            djinn_refl_s = eigen_djinn.scale_scatter(self,flux,djinn_refl_ns,'keep')
            # Combine the two sides
            return sn.pops_robust('scatter',flux.shape,djinn_refl_s,djinn_scatter_s,self.splits)
        return djinn_scatter_s

    def scale_fission(self,phi,djinn_ns):
        import numpy as np
        from discrete1.util import sn
        if np.sum(phi) == 0:
            # return np.zeros((sn.cat(phi,self.splits['fission_djinn']).shape))
            return np.zeros((phi.shape))
        if self.multDJ == 'scatter':
            return np.einsum('ijk,ik->ij',self.chiNuFission,phi) 
        interest = sn.cat(phi,self.splits['fission_djinn'])
        scale = np.sum(interest*np.sum(sn.cat(self.chiNuFission,self.splits['fission_djinn']),axis=1),axis=1)/np.sum(djinn_ns,axis=1)
        # All of the sigma*phi terms not calculated by DJINN
        regular = np.einsum('ijk,ik->ij',sn.cat(self.chiNuFission,self.splits['fission_keep']),sn.cat(phi,self.splits['fission_keep']))
        return sn.pops_robust('fission',phi.shape,regular,scale[:,None]*djinn_ns,self.splits)
        
    def create_fmult(self,flux):
        import numpy as np
        # from discrete1.setup_ke import func

        # model_name = 'fission_1d/carbon/djinn_reg/model_001002'
        
        # _,self.model_fission = func.djinn_load(model_name,self.multDJ)
        
        djinn_fission_ns = 0
        if (np.sum(flux) == 0):
            return np.zeros(flux.shape)
        if self.multDJ in ['both','fission']:
            djinn_fission_ns = eigen_djinn.label_model(self,'fission',flux,self.model_fission)
        return eigen_djinn.scale_fission(self,flux,djinn_fission_ns)

    def one_group(self,total,scatter,djinn_1g,external,guess,tol=1e-08,MAX_ITS=100):
        """ Arguments:
            total: I x 1 vector of the total cross section for each spatial cell
            scatter: I x 1 array for the scattering of the spatial cell by moment
            external: I x N array for the external sources
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
        Returns:
            phi: a I x L+1 array   """
        import numpy as np        
        from discrete1.util import sn
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
            phi = np.zeros((self.I))
            for n in range(self.N):
                weight = self.mu[n]*self.inv_delta
                top_mult = (weight-half_total).astype('float64')
                bottom_mult = (1/(weight+half_total)).astype('float64')
                if (self.multDJ in ['scatter','both']):
                    # temp_scat = sn.pops_robust('scatter',(self.I,),sn.cat(scatter*phi_old,self.splits['scatter_keep']),djinn_1g,self.splits).astype('float64')
                    temp_scat = (djinn_1g).astype('float64')
                else:
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
            
    def multi_group(self,total,scatter,chiNuFission,problem,tol=1e-08,MAX_ITS=100):
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
        from discrete1.setup_ke import func

        # phi_old = func.initial_flux(problem)
        phi_old = problem.copy()
        # phi_old = np.zeros((self.I,self.G))

        converged = 0
        count = 1
        if self.track == 'source':
            allmat_sca = np.zeros((2,0,self.G+1))
        while not (converged):
            phi = np.zeros(phi_old.shape)
            if self.multDJ in ['scatter','both']:
                smult = eigen_djinn.create_smult(self,phi_old)     
                for g in range(self.G):
                    phi[:,g] = eigen_djinn.one_group(self,total[:,g],scatter[:,g,g],smult[:,g],chiNuFission[:,g],phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
                if self.track == 'source':
                    temp_fission,temp_scatter = eigen_djinn.tracking_data(self,phi,chiNuFission)
                    allmat_sca = np.hstack((allmat_sca,temp_scatter))
            elif self.multDJ == 'fission':
                for g in range(self.G):
                    if g == 0:
                        q_tilde = chiNuFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g)
                    else:
                        q_tilde = chiNuFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g) + func.update_q(scatter,phi,0,g,g)
                    phi[:,g] = eigen_djinn.one_group(self,total[:,g],scatter[:,g,g],None,q_tilde,phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
            # phi_old = func.initial_flux(problem)
        if self.track == 'source':
            return phi, allmat_sca
        return phi

    def transport(self,model_name,problem='carbon',multDJ='both',tol=1e-12,MAX_ITS=100,double=False):
        """ EIGEN DJINN SYMM
        Arguments:
            model_name: File location of DJINN model
            problem: used to initialize phi
            multDJ: which DJINN models to use
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a I x G array    """        
        import numpy as np
        from discrete1.setup_ke import func

        phi_old = func.initial_flux(problem)

        self.multDJ = multDJ; self.double = double
        if self.double:
            model_scatter,model_fission,refl_scatter,_ = func.djinn_load_double(model_name,self.multDJ)
            self.refl_scatter = refl_scatter; #self.refl_fission = refl_fission
        else:
            model_scatter,model_fission = func.djinn_load(model_name,self.multDJ)
        self.model_scatter = model_scatter; self.model_fission = model_fission
        
        converged = 0; count = 1
        while not (converged):
            sources = eigen_djinn.create_fmult(self,phi_old)
            print('Outer Transport Iteration {}\n==================================='.format(count))
            if self.track == 'source':
                phi,temp_scatter = eigen_djinn.multi_group(self,self.total,self.scatter,sources,problem,tol=1e-08,MAX_ITS=MAX_ITS)
                enrich = str(np.amax(self.enrich)).split('.')[1]
                np.save('mydata/track_{}_djinn/enrich_{:<02}_count_{}'.format(problem,enrich,str(count).zfill(3)),temp_scatter)
            else:    
                phi = eigen_djinn.multi_group(self,self.total,self.scatter,sources,phi_old,tol=1e-08,MAX_ITS=100)
            keff = np.linalg.norm(phi)
            phi /= keff
            # Check for convergence
            change = np.linalg.norm((phi-phi_old)/phi/(self.I))
            print('Change is',change,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS)
            count += 1
            phi_old = phi.copy()
        return phi,keff 

    def tracking_data(self,phi,sources=None):
        from discrete1.util import sn
        import numpy as np
        # Scatter Tracking - separate phi and add label
        label_scatter = sn.cat(self.enrich,self.splits['scatter_djinn'])
        phi_scatter = sn.cat(phi,self.splits['scatter_djinn'])
        phi_full_scatter = np.hstack((label_scatter[:,None],phi_scatter))
        # Separate scatter multiplier and add label
        multiplier_scatter = sn.cat(np.einsum('ijk,ik->ij',self.scatter,phi),self.splits['scatter_djinn'])
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

class source_djinn:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L,R,I,enrich=None,splits=None,label=None,track=None):
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
        self.enrich = enrich
        self.splits = splits
        self.label = label
        self.track = track
                        
    def label_model(self,xs,flux,model_):
        import numpy as np
        from discrete1.util import sn
        phi = flux.copy()
        if np.sum(phi) == 0:
            return np.zeros((sn.cat(phi,self.splits['{}_djinn'.format(xs)]).shape))
        short_phi = sn.cat(phi,self.splits['{}_djinn'.format(xs)])
        if self.label:
            short_phi = np.hstack((sn.cat(self.enrich,self.splits['{}_djinn'.format(xs)])[:,None],short_phi)) 
        return model_.predict(short_phi)

    def scale_scatter(self,flux,djinn_ns):
        import numpy as np
        from discrete1.util import sn
        phi = flux.copy()
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
        if self.multDJ == 'scatter':
            return np.einsum('ijk,ik->ij',self.chiNuFission,phi) 
        interest = sn.cat(phi,self.splits['fission_djinn'])
        scale = np.sum(interest*np.sum(sn.cat(self.chiNuFission,self.splits['fission_djinn']),axis=1),axis=1)/np.sum(djinn_ns,axis=1)
        # All of the sigma*phi terms not calculated by DJINN
        regular = np.einsum('ijk,ik->ij',sn.cat(self.chiNuFission,self.splits['fission_keep']),sn.cat(phi,self.splits['fission_keep']))
        return sn.pops_robust('fission',phi.shape,regular,scale[:,None]*djinn_ns,self.splits)
    
    def creating_mult(self,flux):
        import numpy as np
        if np.sum(flux) == 0:
            return np.zeros(flux.shape)
        if self.multDJ == 'fission':
            djinn_fission_ns = source_djinn.label_model(self,'fission',flux,self.model_fission)
            fmult = source_djinn.scale_fission(self,flux,djinn_fission_ns)
            # fmult *= fnorm
            smult = np.einsum('ijk,ik->ij',self.scatter,flux)
        elif self.multDJ == 'scatter':
            fmult = np.einsum('ijk,ik->ij',self.chiNuFission,flux)
            djinn_scatter_ns = source_djinn.label_model(self,'scatter',flux,self.model_scatter)
            smult = source_djinn.scale_scatter(self,flux,djinn_scatter_ns)
        elif self.multDJ == 'both':
            djinn_fission_ns = source_djinn.label_model(self,'fission',flux,self.model_fission)
            fmult = source_djinn.scale_fission(self,flux,djinn_fission_ns)
            # fmult *= fnorm
            djinn_scatter_ns = source_djinn.label_model(self,'scatter',flux,self.model_scatter)
            smult = source_djinn.scale_scatter(self,flux,djinn_scatter_ns)
            # smult *= snorm
        return fmult + smult 

    def one_group(self,total,scatter,external):
        """ Arguments:
            total: I x 1 vector of the total cross section for each spatial cell
            scatter: I x 1 array for the scattering of the spatial cell by moment
            external: I x N array for the external sources
        Returns:
            phi: a I x L+1 array   """
        import numpy as np        
        from discrete1.util import sn
        import ctypes
        clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cfunctions.so')
        sweep = clibrary.sweep
        converged = 0
        count = 1
        phi = np.zeros((self.I),dtype='float64')
        half_total = 0.5*total.copy()
        external = external.astype('float64')
        for n in range(self.N):
            weight = self.mu[n]*self.inv_delta
            top_mult = (weight-half_total).astype('float64')
            bottom_mult = (1/(weight+half_total)).astype('float64')
            temp_scat = (scatter).astype('float64')
            # Set Pointers for C function
            phi_ptr = ctypes.c_void_p(phi.ctypes.data)
            ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)
            ext_ptr = ctypes.c_void_p(external.ctypes.data)
            top_ptr = ctypes.c_void_p(top_mult.ctypes.data)
            bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)
            sweep(phi_ptr,ts_ptr,ext_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]))
        return phi

    def transport(self,model_name,problem='carbon',multDJ='both',tol=1e-08,MAX_ITS=1000):
        """ Arguments:
            model_name: File location of DJINN model
            problem: to initialize phi
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            multDJ: which DJINN models to use
        Returns:
            phi: a I x G array  """
        import numpy as np
        from discrete1.setup_ke import func,ex_sources
        
        phi_old = func.initial_flux('carbon_source')
        # phi_old *= 0

        self.multDJ = multDJ
        model_scatter,model_fission = tools.djinn_load(model_name,self.multDJ)
        self.model_scatter = model_scatter; self.model_fission = model_fission

        source = ex_sources.source1(self.I,self.G)

        converged = 0
        count = 1
        while not (converged):
            mult = source_djinn.creating_mult(self,phi_old)
            print('Source Iteration {}'.format(count))
            phi = np.zeros(phi_old.shape)
            for g in range(self.G):
                phi[:,g] = source_djinn.one_group(self,self.total[:,g],mult[:,g],source)            
            # Check for convergence
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            print('Change is',change,'\n===================================')
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi      