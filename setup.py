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
        if problem == 'mixed1':
            # np.load('discrete1/data/phi_{}_15.npy'.format(problem))
            problem = 'carbon'
        return np.load('discrete1/data/phi_{}_15.npy'.format(problem))
#        if problem == 'carbon':
#            return np.load('discrete1/data/phi_orig_15.npy')
#        elif problem == 'stainless':
#            return np.load('discrete1/data/phi_ss_15.npy')
#        elif problem == 'stainless_flip':
#            return np.load('discrete1/data/phi_ss_flip_15.npy')
#        elif problem == 'multiplastic':
#            return np.load('discrete1/data/phi_mp_15.npy')
#        elif problem == 'mixed1':
#            return np.load('discrete1/data/phi_mixed1.npy')
#        elif problem == 'carbon_source':
#            return np.load('discrete1/data/phi_carbon_source.npy')

    def low_rank_svd(phi,scatter,fission,problem,rank,distance):
        import numpy as np
        from discrete1.util import sn
        from discrete1.setup import problem1
        # List of the material splits
        mat_split = problem1.boundaries_mat(problem,distance=distance)
        # New Matrices to put scatter and fission matrices
        r_scatter = np.empty(scatter.shape)
        r_fission = np.empty(fission.shape)

        for mat in mat_split:
            # SVD of Phi
            phi_ = phi[mat].T.copy()
            u_p,_,_ = sn.svd(phi_,rank)
            # SVD of Scatter
            smult_ = scatter[mat][0] @ phi_
            u_s,_,_ = sn.svd(smult_,rank)
            # SVD of Fission
            fmult_ = fission[mat][0] @ phi_    
            u_f,_,_ = sn.svd(smult_,rank)
            # Reduce and resize Scatter
            s_tilde = u_s.T @ scatter[mat][0] @ u_p
            ys_tilde = u_s @ s_tilde @ u_p.T
            # Reduce and resize Fission
            f_tilde = u_f.T @ fission[mat][0] @ u_p
            yf_tilde = u_f @ f_tilde @ u_p.T
            # Repopulate Scatter and Fission
            r_scatter[mat] = np.tile(ys_tilde,(sn.length(mat),1,1))
            r_fission[mat] = np.tile(yf_tilde,(sn.length(mat),1,1))

        return r_scatter,r_fission
    
    # def normalize(data,verbose=False):
    #     import numpy as np
    #     maxi = np.amax(data,axis=1)
    #     mini = np.amin(data,axis=1)
    #     norm = (data-mini[:,None])/(maxi-mini)[:,None]
    #     if verbose:
    #         return norm,maxi,mini
    #     return norm
    
    # def unnormalize(data,maxi,mini):
    #     return data*(maxi-mini)[:,None]+mini[:,None]

    def djinn_load(model_name,dtype):
        from djinn import djinn
        # from dj2.djinn import djinn
        if dtype == 'both':
            model_scatter = djinn.load(model_name=model_name[0])
            model_fission = djinn.load(model_name=model_name[1])
        elif dtype == 'scatter':
            model_scatter = djinn.load(model_name=model_name)
            model_fission = None
        elif dtype == 'fission':
            model_scatter = None
            model_fission = djinn.load(model_name=model_name)
        return model_scatter,model_fission

    def load_coder(coder,ptype='phi'):
        """ Coder is the string path to the autoencoder, encoder, and decoder """
        from tensorflow import keras
        if coder == 'dummy1':
            return func.load_dummy1(),func.load_dummy1(),func.load_dummy1()
        elif coder == 'dummy2':
            return func.load_dummy2(),func.load_dummy2(),func.load_dummy2()
        elif coder == 'dummy3':
            return func.load_dummy3(),func.load_dummy3(),func.load_dummy3()
        print('Loading {} autoencoder...'.format(ptype))
        autoencoder = keras.models.load_model('{}_{}_autoencoder.h5'.format(coder,ptype))
        encoder = keras.models.load_model('{}_{}_encoder.h5'.format(coder,ptype),compile=False)
        decoder = keras.models.load_model('{}_{}_decoder.h5'.format(coder,ptype),compile=False)
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

class problem2:
    def variables(dim=70,distance=[5,5,10,5,5]):
        import numpy as np
        from discrete1.util import sn
        I = 1000
        delta = sum(distance)/I
        # Scattering
        pu240_scatter = np.load('mydata/pu240/scatter_{}.npy'.format(str(dim).zfill(3)))
        puc240_scatter = np.load('mydata/puc240/scatter_{}.npy'.format(str(dim).zfill(3)))
        # Fission
        pu240_fission = np.load('mydata/pu240/nu_fission_{}.npy'.format(str(dim).zfill(3)))
        puc240_fission = np.load('mydata/puc240/nu_fission_{}.npy'.format(str(dim).zfill(3)))
        # Total
        pu240_total = np.load('mydata/pu240/total_{}.npy'.format(str(dim).zfill(3)))
        puc240_total = np.load('mydata/puc240/total_{}.npy'.format(str(dim).zfill(3)))
        # Ordering
        xs_scatter = [puc240_scatter,pu240_scatter,puc240_scatter,pu240_scatter,puc240_scatter]
        # xs_scatter = [puc240_scatter.T,pu240_scatter.T,puc240_scatter.T,pu240_scatter.T,puc240_scatter.T]
        xs_total = [puc240_total,pu240_total,puc240_total,pu240_total,puc240_total]
        xs_fission = [puc240_fission,pu240_fission,puc240_fission,pu240_fission,puc240_fission]
        # xs_fission = [puc240_fission.T,pu240_fission.T,puc240_fission.T,pu240_fission.T,puc240_fission.T]
        
        # Setting up eigenvalue equation
        N = 8; L = 0; R = sum(distance); G = dim
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w)
        # Half because symmetry
        mu = mu[int(N*0.5):]
        w = w[int(N*0.5):]
        N = int(N*0.5) 

        layers = [int(ii/delta) for ii in distance]
        # Use for layer correction
        if sum(layers) != sum(distance):
            change = I - sum(layers)
            if change % 2 != 0:
                layers[2] = layers[2] + 1
                change -= 1
            layers[0] = layers[0] + int(0.5*change)
            layers[-1] = layers[-1] + int(0.5*change)
        # I = int(sum(layers))
    
        scatter_ = sn.mixed_propagate(xs_scatter,layers,G=dim,L=L,dtype='scatter')
        fission_ = sn.mixed_propagate(xs_fission,layers,G=dim,dtype='fission2')
        total_ = sn.mixed_propagate(xs_total,layers,G=dim)
        
        return G,N,mu,w,total_,scatter_[:,0],fission_,L,R,I

    # def boundaries(dim=70,distance=[5,5,10,5,5]):
    #     problem_scatter = problem + '_full'
    #     # Set Fission Splits
    #     enrichment,splits = problem1.boundaries_aux(conc,problem,distance)
    #     fission_splits = {f'fission_{kk}': vv for kk, vv in splits.items()}
    #     # Set Scatter Splits
    #     enrichment,splits = problem1.boundaries_aux(conc,problem_scatter,distance)
    #     scatter_splits = {f'scatter_{kk}': vv for kk, vv in splits.items()}
    #     combo_splits = {**scatter_splits, **fission_splits}
    #     return enrichment,combo_splits

    def scatter_fission(dim=70,distance=[5,5,10,5,5]):
        _,_,_,_,_,scatter,fission,_,_,_ = problem2.variables(dim,distance)
        return scatter,fission


class problem1:        
    def variables(conc=None,problem=None,distance=[45,35,20]):
        from discrete1.util import chem,sn
        import numpy as np
        if problem == 'stainless' or problem == 'carbon':
            pass
            # distance = [45,35,20]
            # distance = [40,40,20]
        elif problem == 'multiplastic':
            distance = [10]*8; distance.append(20)
        elif problem == 'mixed1':
            distance = [45,5,25,5,20]
            conc = [0.12,0.27]
        elif problem == 'noplastic':
            distance = [35,20]
        # elif problem == 'stainless_flip':
        #     distance = [20,35,45]
        delta = 0.1
        if conc is None:
            conc = 0.2
        print('Concentration: ',conc)
        # Layer densities
        density_uh3 = 10.95; density_ch3 = 0.97
        if problem == 'mixed1':
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
        if problem == 'mixed1':
            uh3_scatter_low = uh3_density_low[0]*u235scatter + uh3_density_low[1]*u238scatter + uh3_density_low[2]*h1scatter
        hdpe_scatter = hdpe_density[0]*c12scatter + hdpe_density[1]*h1scatter
        uh3_238_scatter = uh3_238_density[0]*u238scatter + uh3_238_density[1]*h1scatter
    
        # Total Cross Section
        u235total = np.load('mydata/u235/vecTotal.npy')[eval(spec_temp)]
        u238total = np.load('mydata/u238/vecTotal.npy')[eval(spec_temp)]
        h1total = np.load('mydata/h1/vecTotal.npy')[eval(spec_temp)]
        c12total = np.load('mydata/cnat/vecTotal.npy')[eval(spec_temp)]
    
        uh3_total = uh3_density[0]*u235total + uh3_density[1]*u238total + uh3_density[2]*h1total
        if problem == 'mixed1':
            uh3_total_low = uh3_density_low[0]*u235total + uh3_density_low[1]*u238total + uh3_density_low[2]*h1total
        hdpe_total = hdpe_density[0]*c12total + hdpe_density[1]*h1total
        uh3_238_total = uh3_238_density[0]*u238total + uh3_238_density[1]*h1total
    
        # Fission Cross Section
        u235fission = np.load('mydata/u235/nufission_0{}.npy'.format(spec_temp))[0]
        u238fission = np.load('mydata/u238/nufission_0{}.npy'.format(spec_temp))[0]
    
        uh3_fission = uh3_density[0]*u235fission + uh3_density[1]*u238fission
        if problem == 'mixed1':
            uh3_fission_low = uh3_density_low[0]*u235fission + uh3_density_low[1]*u238fission
        uh3_238_fission = uh3_238_density[0]*u238fission
        hdpe_fission = np.zeros((dim,dim))
    
        # Cross section layers
        if problem == 'carbon':
            xs_scatter = [hdpe_scatter.T,uh3_scatter.T,uh3_238_scatter.T]
            xs_total = [hdpe_total,uh3_total,uh3_238_total]
            xs_fission = [hdpe_fission.T,uh3_fission.T,uh3_238_fission.T]
        elif problem == 'mixed1':
            xs_scatter = [hdpe_scatter.T,uh3_scatter_low.T,uh3_scatter.T,uh3_scatter_low.T,uh3_238_scatter.T]
            xs_total = [hdpe_total,uh3_total_low,uh3_total,uh3_total_low,uh3_238_total]
            xs_fission = [hdpe_fission.T,uh3_fission_low.T,uh3_fission.T,uh3_fission_low.T,uh3_238_fission.T]
        elif problem == 'multiplastic':
            xs_scatter = []; xs_total = []; xs_fission = []
            for ii in range(4):
                xs_scatter.append(hdpe_scatter.T); xs_scatter.append(uh3_scatter.T)
                xs_total.append(hdpe_total); xs_total.append(uh3_total)
                xs_fission.append(hdpe_fission.T); xs_fission.append(uh3_fission.T)
            xs_scatter.append(uh3_238_scatter.T)
            xs_total.append(uh3_238_total)
            xs_fission.append(uh3_238_fission.T)
        elif problem == 'stainless':
            print('Using Stainless Steel')
            ss_total,ss_scatter = chem.xs_ss440(dim, spec_temp)
            xs_scatter = [ss_scatter.T,uh3_scatter.T,uh3_238_scatter.T]
            xs_total = [ss_total,uh3_total,uh3_238_total]
            xs_fission = [hdpe_fission.T,uh3_fission.T,uh3_238_fission.T]
        elif problem == 'noplastic':
            xs_scatter = [uh3_scatter.T,uh3_238_scatter.T]
            xs_total = [uh3_total,uh3_238_total]
            xs_fission = [uh3_fission.T,uh3_238_fission.T]
        # Setting up eigenvalue equation
        N = 8; L = 0; R = sum(distance); G = dim
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w)
        # Half because symmetry
        mu = mu[int(N*0.5):]
        w = w[int(N*0.5):]
        N = int(N*0.5) 

        layers = [int(ii/delta) for ii in distance]
        I = int(sum(layers))
    
        scatter_ = sn.mixed_propagate(xs_scatter,layers,G=dim,L=L,dtype='scatter')
        fission_ = sn.mixed_propagate(xs_fission,layers,G=dim,dtype='fission2')
        total_ = sn.mixed_propagate(xs_total,layers,G=dim)
        
        return G,N,mu,w,total_,scatter_[:,0],fission_,L,R,I
    
    def boundaries_aux(conc,problem=None,distance=[45,35,20]):
        import numpy as np
        from discrete1.util import sn
        # Fission Models
        if problem == 'carbon' or problem == 'stainless':
            # distance = [45,35,20]; 
            ment = [conc,0]; where = [1,2]
        elif problem ==  'mixed1':
            distance = [45,5,25,5,20]; ment = [0.12,0.27,0.12,0]; where = [1,2,3,4]
        elif problem == 'multiplastic':
            distance = [10]*8; distance.append(20); ment = [conc]*4; ment.append(0); where = [1,3,5,7,8]

        # elif problem == 'blur':
        #     distance = [47,33,20]; ment = [conc,0]; where = [1,2]
        # elif problem == 'noplastic':
        #     distance = [35,20]; ment = [conc,0]; where = [0,1]
        # elif problem == 'stainless_flip':
        #     distance = [20,35,45]; ment = [0,conc]; where = [0,1]

        # Scatter Models
        elif problem == 'carbon_full':
            # distance = [45,35,20]; 
            ment = [15.04,conc,0]; where = [0,1,2]
        elif problem == 'mixed1_full':
            distance = [45,5,25,5,20]; ment = [15.04,0.12,0.27,0.12,0]; where = [0,1,2,3,4]
        elif problem == 'multiplastic_full':
            distance = [10]*8; distance.append(20); ment = [15.04,conc]*4; ment.append(0); where = [0,1,2,3,4,5,6,7,8]
        elif problem == 'stainless_full':
            # distance = [45,35,20]; 
            ment = [52.68,conc,0]; where = [0,1,2]
        delta = 0.1
        layers = [int(ii/delta) for ii in distance]
        splits = np.array(sn.layer_slice(layers))
        # conc is uh3 enrich while 0 is depleted uranium
        enrichment = sn.enrich_list(sum(layers),ment,splits[where].tolist())
        return enrichment,sn.layer_slice_dict(layers,where)

    def boundaries_mat(problem,distance=[45,35,20]):
        import numpy as np
        from discrete1.util import sn
        # distance = [45,35,20]
        if problem ==  'mixed1':
            distance = [45,5,25,5,20]
        elif problem == 'multiplastic':
            distance = [10]*8; distance.append(20)
        delta = 0.1
        layers = [int(ii/delta) for ii in distance]
        return np.sort(np.array(sn.layer_slice(layers)))
    
    def boundaries(conc=0.2,problem=None,distance=[45,35,20]):
        problem_scatter = problem + '_full'
        # Set Fission Splits
        enrichment,splits = problem1.boundaries_aux(conc,problem,distance)
        fission_splits = {f'fission_{kk}': vv for kk, vv in splits.items()}
        # Set Scatter Splits
        enrichment,splits = problem1.boundaries_aux(conc,problem_scatter,distance)
        scatter_splits = {f'scatter_{kk}': vv for kk, vv in splits.items()}
        combo_splits = {**scatter_splits, **fission_splits}
        return enrichment,combo_splits

    def scatter_fission(conc,problem,distance=[45,35,20]):
        _,_,_,_,_,scatter,fission,_,_,_ = problem1.variables(conc,problem,distance)
        return scatter,fission


class ex_sources:
    def source1(I,G):
        """ One Unit Source in 14.1 MeV group from left"""
        import numpy as np
        from discrete1.util import display
        energy = display.gridPlot()
        g = np.argmin(abs(energy-14.1E6))
        source = np.zeros((I,G))
        source[0,g] = 1
        return source
