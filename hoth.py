class func:
    def __init__(self):
        self.dum1 = self.load_dummy1()
        self.dum2 = self.load_dummy2()
        self.dum3 = self.load_dummy3()
        
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

    def initial_flux(problem):
        import numpy as np
        if problem == 'carbon':
            return np.load('discrete1/data/phi_orig_15.npy')
        elif problem == 'stainless':
            return np.load('discrete1/data/phi_ss_15.npy')
        elif problem == 'stainless_flip':
            return np.load('discrete1/data/phi_ss_flip_15.npy')
        elif problem == 'multiplastic':
            return np.load('discrete1/data/phi_mp_15.npy')
        elif problem == 'mixed1':
            return np.load('discrete1/data/phi_mixed1.npy')
    
    def normalize(data,verbose=False):
        import numpy as np
        maxi = np.amax(data,axis=1)
        mini = np.amin(data,axis=1)
        norm = (data-mini[:,None])/(maxi-mini)[:,None]
        if verbose:
            return norm,maxi,mini
        return norm
    
    def unnormalize(data,maxi,mini):
        return data*(maxi-mini)[:,None]+mini[:,None]

    def load_coder(coder):
        """ Coder is the string path to the autoencoder, encoder, and decoder """
        from tensorflow import keras
        if coder == 'dummy1':
            return func.load_dummy1(),None,None
        elif coder == 'dummy2':
            return func.load_dummy2(),None,None
        elif coder == 'dummy3':
            return func.load_dummy3(),None,None

        autoencoder = keras.models.load_model('{}_autoencoder.h5'.format(coder))
        encoder = keras.models.load_model('{}_encoder.h5'.format(coder),compile=False)
        decoder = keras.models.load_model('{}_decoder.h5'.format(coder),compile=False)
        return autoencoder,encoder,decoder

    def find_gprime(coder):
        import re
        # model = coder.split('model')[1]
        nums = re.findall(r'\d+',coder.split('model')[1])
        return min([int(ii) for ii in nums])

    class load_dummy1:
        # Original
        def predict(self,array):
            return array
        
    class load_dummy2:
        # Half
        def predict(self,array):
            return 0.5*array
    
    class load_dummy3:
        # Random Noise
        def predict(self,array):
            import numpy as np
            return array + (0.001*np.random.normal(0,1,array.shape[0]))[:,None]


class problem:        
    def variables(conc=None,ptype=None,distance=None,symm=False):
        from discrete1.util import chem,sn
        import numpy as np
        if ptype is None or ptype == 'stainless' or ptype == 'carbon_full' or ptype == 'carbon':
            distance = [45,35,20]
        elif ptype == 'multiplastic':
            distance = [10]*8; distance.append(20)
        elif ptype == 'mixed1':
            distance = [45,5,25,5,20]
            conc = [0.12,0.27]
        elif ptype == 'noplastic':
            distance = [35,20]
        elif ptype == 'stainless_flip':
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
        elif ptype == 'stainless':
            print('Using Stainless Steel')
            ss_total,ss_scatter = chem.xs_ss440(dim, spec_temp)
            xs_scatter = [ss_scatter.T,uh3_scatter.T,uh3_238_scatter.T]
            xs_total = [ss_total,uh3_total,uh3_238_total]
            xs_fission = [hdpe_fission.T,uh3_fission.T,uh3_238_fission.T]
        # elif ptype == 'stainless_flip':
        #     print('Using FLIPPED Stainless Steel')
        #     ss_total,ss_scatter = chem.xs_ss440(dim, spec_temp)
        #     xs_scatter = [uh3_238_scatter.T,uh3_scatter.T,ss_scatter.T]
        #     xs_total = [uh3_238_total,uh3_total,ss_total]
        #     xs_fission = [uh3_238_fission.T,uh3_fission.T,hdpe_fission.T]
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
        if ptype == 'carbon' or ptype == 'stainless':
            distance = [45,35,20]; ment = [conc,0]; where = [1,2]
        elif ptype == 'multiplastic':
            distance = [10]*8; distance.append(20); ment = [conc]*4; ment.append(0); where = [1,3,5,7,8]
        elif ptype == 'multiplastic_full':
            distance = [10]*8; distance.append(20); ment = [15.04,conc]*4; ment.append(0); where = [0,1,2,3,4,5,6,7,8]
        elif ptype ==  'mixed1':
            distance = [45,5,25,5,20]; ment = [0.12,0.27,0.12,0]; where = [1,2,3,4]
        elif ptype == 'mixed1_full':
            distance = [45,5,25,5,20]; ment = [15.04,0.12,0.27,0.12,0]; where = [0,1,2,3,4]
        elif ptype == 'blur':
            distance = [47,33,20]; ment = [conc,0]; where = [1,2]
        elif ptype == 'noplastic':
            distance = [35,20]; ment = [conc,0]; where = [0,1]
        elif ptype == 'stainless_flip':
            distance = [20,35,45]; ment = [0,conc]; where = [0,1]
        elif ptype == 'carbon_full':
            distance = [45,35,20]; ment = [15.04,conc,0]; where = [0,1,2]
        elif ptype == 'stainless_full':
            distance = [45,35,20]; ment = [52.68,conc,0]; where = [0,1,2]
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
            # smult = eigen_auto.update_q(self,scatter,phi_old)

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
                phi[:,g] = eigen_auto.one_group(self,total[:,g],smult[:,g],nuChiFission[:,g],tol=tol,MAX_ITS=MAX_ITS,guess=phi_old[:,g])
            # print(phi.shape,phi_old.shape,phi_old_full.shape)
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi
            
    # def shrink(self,phi):
    #     from discrete1.util import nnets
    #     # Normalize, save maxi and mini
    #     norm_phi,maxi,mini = nnets.normalize(phi,verbose=True)
    #     self.maxi = maxi; self.mini = mini
    #     # Encode to G'
    #     return self.encoder.predict(norm_phi)

    # def grow(self,phi,norm=False):
    #     from discrete1.util import nnets
    #     # Decode out to G 
    #     phi_full = self.decoder.predict(phi)
    #     # Unnormalize
    #     phi_full = nnets.unnormalize(phi_full,self.maxi,self.mini)
    #     # Multiply by XS
    #     if norm:
    #         import numpy as np
    #         keff = np.linalg.norm(phi_full)
    #         phi_full /= keff
    #         return phi_full,keff
    #     return phi_full

    # def update_q(self,xs,phi,small=True):
    #     import numpy as np
    #     from discrete1.util import nnets
    #     if small:
    #         phi_full = eigen_auto.grow(self,phi)
    #     else:
    #         phi_full = phi.copy()
    #     product = np.einsum('ijk,ik->ij',xs,phi_full)
    #     return eigen_auto.shrink(self,product)
        

    def transport(self,coder,problem='carbon',tol=1e-12,MAX_ITS=100,LOUD=True):
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
        # sources = eigen_auto.update_q(self,self.chiNuFission,phi_old,small=False)
        # phi_old_small = eigen_auto.shrink(self,phi_old) 
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
            phi = eigen_auto.multi_group(self,self.total,self.scatter,sources,phi_old,tol=1e-08,MAX_ITS=MAX_ITS)
            # Convert to phi G, normalized
            # phi,keff = eigen_auto.grow(self,phi,norm=True)
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
            # sources = eigen_auto.update_q(self,self.chiNuFission,phi_old,small=False)
            # phi_old_small = eigen_auto.shrink(self,phi_old) 
        return phi,keff

class data_collect:
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
        clibrary = ctypes.cdll.LoadLibrary('./discrete1/cfunctions.so')
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
                phi[:,g] = data_collect.one_group(self,total[:,g],smult[:,g],nuChiFission[:,g],phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
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
                phi,allmat_phi = data_collect.multi_group(self,self.total,self.scatter,sources,phi_old,tol=1e-08,MAX_ITS=MAX_ITS)
                np.save('mydata/track_ae_{}/phi_{:<02}_count_{}'.format(problem,enrich,str(count).zfill(3)),allmat_phi)
            else:
                phi = data_collect.multi_group(self,self.total,self.scatter,sources,phi_old,tol=1e-08,MAX_ITS=MAX_ITS)
            
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

class test:
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
        clibrary = ctypes.cdll.LoadLibrary('./discrete1/cfunctions.so')
        sweep = clibrary.sweep
        converged = 0
        count = 1        
        phi = np.zeros((self.I),dtype='float64')
        phi_old = guess.copy()
        # phi_full = guess.copy()
        half_total = 0.5*total.copy()
        external = external.astype('float64')
        # while not(converged):
            # phi *= 0
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
                
                # psi_bottom = 0 # vacuum on LHS
                # # Left to right
                # for ii in range(self.I):
                #     psi_top = (temp_scat[ii] + external[ii] + psi_bottom * (weight-half_total[ii]))/(weight+half_total[ii])
                #     phi[ii] = phi[ii] + (self.w[n] * func.diamond_diff(psi_top,psi_bottom))
                #     psi_bottom = psi_top
                # # Reflective right to left
                # for ii in range(self.I-1,-1,-1):
                #     psi_top = psi_bottom
                #     psi_bottom = (temp_scat[ii] + external[ii] + psi_top * (weight-half_total[ii]))/(weight+half_total[ii])
                #     phi[ii] = phi[ii] +  (self.w[n] * func.diamond_diff(psi_top,psi_bottom))
            # change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            # converged = (change < tol) or (count >= MAX_ITS) 
            # count += 1
            # phi_old = phi.copy()
            # phi_full[:,g] = phi_old.copy()
        return phi

    def multi_group(self,total,scatter,nuChiFission,guess,tol=1e-08,MAX_ITS=10000,normal=True):
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
        phi_old_full = guess.copy()
        phi_old = guess.copy()
        
        smult = np.einsum('ijk,ik->ij',scatter,phi_old)

        converged = 0
        count = 1

        while not (converged):
            phi = np.zeros(phi_old.shape)  
            for g in range(self.gprime):
                phi[:,g] = test.one_group(self,total[:,g],smult[:,g],nuChiFission[:,g],phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
            
            if normal:
                phi_full,pmaxi,pmini = nnets.normalize(phi,verbose=True)
                scale = np.sum(phi_full,axis=1)
                phi_full = self.autoencoder.predict(phi_full)
                phi_full = (scale/np.sum(phi_full,axis=1))[:,None]*phi_full
                phi_full = nnets.unnormalize(phi_full,pmaxi,pmini)
            else:
                phi_full = self.autoencoder.predict(phi)
                phi_full = (np.sum(phi,axis=1)/np.sum(phi_full,axis=1))[:,None]*phi_full
            
            change = np.linalg.norm((phi_full - phi_old_full)/phi_full/(self.I))
            count += 1

            converged = (change < tol) or (count >= MAX_ITS) 
            phi_old_full = phi_full.copy()
            # phi_old = phi_old_full.copy()
            phi_old = phi.copy()

            smult = np.einsum('ijk,ik->ij',scatter,phi_full)

            # smult,smaxi,smini = nnets.normalize(smult,verbose=True)
            # smult[np.isnan(smult)] = 0
            # smaxi[np.isnan(smaxi)] = 0; smini[np.isnan(smini)] = 0
            # smult = self.autoencoder.predict(smult)
            # smult = nnets.unnormalize(smult,smaxi,smini)

            # Scaling for Multiplication
            # scale = np.sum(phi_full*np.sum(scatter,axis=1),axis=1)/np.sum(smult,axis=1)
            # smult = scale[:,None]*smult

        return phi_full
            
    def transport(self,coder,problem='carbon',tol=1e-12,MAX_ITS=100,LOUD=True,normal=True):
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
        self.gprime = 87
        # self.encoder = encoder; self.decoder = decoder
        self.autoencoder = autoencoder

        phi_old_full = func.initial_flux(problem)
        keff = np.linalg.norm(phi_old_full)
        if normal:
            phi_old,pmaxi,pmini = nnets.normalize(phi_old_full,verbose=True)
            scale = np.sum(phi_old,axis=1)
            phi_old = self.autoencoder.predict(phi_old)
            phi_old = (scale/np.sum(phi_old,axis=1))[:,None]*phi_old
            phi_old = nnets.unnormalize(phi_old,pmaxi,pmini)
        else:
            phi_old = self.autoencoder.predict(phi_old_full)
            phi_old = (np.sum(phi_old_full,axis=1)/np.sum(phi_old,axis=1))[:,None]*phi_old

        converged = 0
        count = 1

        sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old)

        # sources,fmaxi,fmini = nnets.normalize(sources,verbose=True)
        # sources[np.isnan(sources)] = 0;
        # fmaxi[np.isnan(fmaxi)] = 0; fmini[np.isnan(fmini)] = 0
        # sources = self.autoencoder.predict(sources)
        # sources = nnets.unnormalize(sources,fmaxi,fmini)
        # Scaling Matrix Multiplication
        # scale = np.sum(sources_old,axis=1)/np.sum(sources,axis=1)
        # scale[np.isnan(scale)] = 0;
        # sources = scale[:,None]*sources

        while not (converged):
            print('Outer Transport Iteration {}'.format(count))
            phi = test.multi_group(self,self.total,self.scatter,sources,phi_old,tol=1e-08,MAX_ITS=MAX_ITS,normal=normal)

            if normal:
                phi_full,pmaxi,pmini = nnets.normalize(phi,verbose=True)
                scale = np.sum(phi_full,axis=1)
                phi_full = self.autoencoder.predict(phi_full)
                phi_full = (scale/np.sum(phi_full,axis=1))[:,None]*phi_full
                phi_full = nnets.unnormalize(phi_full,pmaxi,pmini)
            else:
                phi_full = self.autoencoder.predict(phi)
                phi_full = (np.sum(phi,axis=1)/np.sum(phi_full,axis=1))[:,None]*phi_full
            
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

            sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old_full)
            
            # sources,fmaxi,fmini = nnets.normalize(sources,verbose=True)
            # sources[np.isnan(sources)] = 0;
            # fmaxi[np.isnan(fmaxi)] = 0; fmini[np.isnan(fmini)] = 0
            # sources = self.autoencoder.predict(sources)
            # sources = nnets.unnormalize(sources,fmaxi,fmini)
            # # Scaling Matrix Multiplication
            # scale = np.sum(phi_old_full*np.sum(self.chiNuFission,axis=1),axis=1)/np.sum(sources,axis=1)
            # scale[np.isnan(scale)] = 0;
            # sources = scale[:,None]*sources

        return phi_full,keff


class source_auto:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L,R,I,track=False):
        self.G = G
        self.N = N
        self.mu = mu
        self.w = w
        self.total = total
        self.scatter = scatter
        # self.chiNuFission = chiNuFission
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
        clibrary = ctypes.cdll.LoadLibrary('./discrete1/cfunctions.so')
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

    def source1(self):
        """ One Unit Source in 14.1 MeV group from left"""
        import numpy as np
        from discrete1.util import display
        energy = display.gridPlot()
        g = np.argmin(abs(energy-14.1E6))
        source = np.zeros((self.I,self.G))
        source[0,g] = 1
        return source

    def transport(self,coder,problem='carbon',tol=1e-08,MAX_ITS=10000,normal=True):
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
        autoencoder,encoder,decoder = func.load_coder(coder)
        self.gprime = 87
        # self.encoder = encoder; self.decoder = decoder
        self.autoencoder = autoencoder

        phi_old_full = func.initial_flux(problem)
        if normal:
            phi_old,pmaxi,pmini = nnets.normalize(phi_old_full,verbose=True)
            scale = np.sum(phi_old,axis=1)
            phi_old = self.autoencoder.predict(phi_old)
            phi_old = (scale/np.sum(phi_old,axis=1))[:,None]*phi_old
            phi_old = nnets.unnormalize(phi_old,pmaxi,pmini)
        else:
            phi_old = self.autoencoder.predict(phi_old_full)
        # Scaling Flux
        phi_old = (np.sum(phi_old_full,axis=1)/np.sum(phi_old,axis=1))[:,None]*phi_old

        smult = np.einsum('ijk,ik->ij',self.scatter,phi_old)
        source = source_auto.source1(self)

        converged = 0
        count = 1

        while not (converged):
            print('Source Iteration {}'.format(count))
            phi = np.zeros(phi_old.shape)  
            for g in range(self.gprime):
                phi[:,g] = source_auto.one_group(self,self.total[:,g],smult[:,g],source,phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
            
            if normal:
                phi_full,pmaxi,pmini = nnets.normalize(phi,verbose=True)
                scale = np.sum(phi_full,axis=1)
                phi_full = self.autoencoder.predict(phi_full)
                phi_full = (scale/np.sum(phi_full,axis=1))[:,None]*phi_full
                phi_full = nnets.unnormalize(phi_full,pmaxi,pmini)
            else:
                phi_full = self.autoencoder.predict(phi)
            # Scaling Factor
            phi_full = (np.sum(phi,axis=1)/np.sum(phi_full,axis=1))[:,None]*phi_full
            
            change = np.linalg.norm((phi_full - phi_old_full)/phi_full/(self.I))
            if LOUD:
                print('Change is',change)
                print('===================================')
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 

            phi_old_full = phi_full.copy()
            phi_old = phi.copy()

            smult = np.einsum('ijk,ik->ij',self.scatter,phi_full)

        return phi_full
            
class source_collect:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L,R,I,track=False):
        self.G = G
        self.N = N
        self.mu = mu
        self.w = w
        self.total = total
        self.scatter = scatter
        # self.chiNuFission = chiNuFission
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
        clibrary = ctypes.cdll.LoadLibrary('./discrete1/cfunctions.so')
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

    def source1(self):
        """ One Unit Source in 14.1 MeV group from left"""
        import numpy as np
        from discrete1.util import display
        energy = display.gridPlot()
        g = np.argmin(abs(energy-14.1E6))
        source = np.zeros((self.I,self.G))
        source[0,g] = 1
        return source

    def transport(self,coder,problem='carbon',tol=1e-08,MAX_ITS=10000,normal=True):
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
        
        phi_old = func.initial_flux(problem)

        smult = np.einsum('ijk,ik->ij',self.scatter,phi_old)
        source = source_auto.source1(self)

        converged = 0
        count = 1
        if self.track == 'source':
            allmat_phi = []
        while not (converged):
            print('Source Iteration {}'.format(count))
            phi = np.zeros(phi_old.shape)  
            for g in range(self.G):
                phi[:,g] = source_auto.one_group(self,self.total[:,g],smult[:,g],source,phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS)
            if self.track == 'source':
                allmat_phi.append(phi)
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            print('Change is',change,'\n===================================')
                
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 

            phi_old = phi.copy()

            smult = np.einsum('ijk,ik->ij',self.scatter,phi)
        if self.track == 'source':
            return phi,np.concatenate((allmat_phi))
        return phi
