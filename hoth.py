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

    def initial_flux(problem):
        if problem == 'carbon':
            return np.load('discrete1/data/phi_orig_15.npy')
        elif probem == 'stainless':
            return np.load('discrete1/data/phi_ss_15.npy')
        elif problem == 'stainless_flip':
            return np.load('discrete1/data/phi_ss_flip_15.npy')
        elif problem == 'multiplastic':
            return np.load('discrete1/data/phi_mp_15.npy')
        elif problem == 'mixed1':
            return np.load('discrete1/data/phi_mixed1.npy')
        
    def load_coder(coder):
        """ Coder is the string path to the autoencoder, encoder, and decoder """
        from tensorflow import keras
        autoencoder = keras.models.load_model('{}_autoencoder.h5'.format(coder))
        encoder = keras.models.load_model('{}_encoder.h5'.format(coder))
        decoder = keras.models.load_model('{}_decoder.h5'.format(coder))
        return autoencoder,encoder,decoder

    def find_gprime(coder):
        import re
        # model = coder.split('model')[1]
        nums = re.findall(r'\d+',coder.split('model')[1])
        return min([int(ii) for ii in nums])
    
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
            distance = [45,35,20]
            ment = [conc,0]
            where = [1,2]
        elif ptype == 'multiplastic':
            distance = [10]*8; distance.append(20)
            ment = [conc]*4; ment.append(0)
            where = [1,3,5,7,8]
        elif ptype == 'multiplastic_full':
            distance = [10]*8; distance.append(20)
            ment = [15.04,conc]*4; ment.append(0)
            where = [0,1,2,3,4,5,6,7,8]
        elif ptype ==  'mixed1':
            distance = [45,5,25,5,20]
            ment = [0.12,0.27,0.12,0]
            where = [1,2,3,4]
        elif ptype == 'mixed1_full':
            distance = [45,5,25,5,20]
            ment = [15.04,0.12,0.27,0.12,0]
            where = [0,1,2,3,4]
        elif ptype == 'blur':
            distance = [47,33,20]
            ment = [conc,0]
            where = [1,2]
        elif ptype == 'noplastic':
            distance = [35,20]
            ment = [conc,0]
            where = [0,1]
        elif ptype == 'stainless_flip':
            distance = [20,35,45]
            ment = [0,conc]
            where = [0,1]
        elif ptype == 'carbon_full' or ptype == 'stainless_full':
            distance = [45,35,20]
            ment = [15.04,conc,0]
            where = [0,1,2]
        elif ptype == 'stainless_full':
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
            
    def transport(self,coder,problem='carbon',tol=1e-12,MAX_ITS=100,LOUD=True):
        """ Arguments:
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a I x G array    """        
        import numpy as np
        # from discrete1.util import sn
        autoencoder,encoder,decoder = func.load_coder(coder)
        self.gprime = func.find_gprime(coder)

        phi_old = func.initial_flux(problem)
        k_old = np.linalg.norm(phi_old)
        track_k = [k_old]


        converged = 0
        count = 1
        sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old) 
        sources = encoder.predict(sources) # reduce down to G'
        while not (converged):
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
            sources = np.einsum('ijk,ik->ij',self.chiNuFission,decoder.predict(phi_old)) 
            sources = encoder.predict(sources)
        return phi,keff
    