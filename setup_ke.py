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

    def initial_flux(problem,group=618):
        import numpy as np
        if problem in ['mixed1','hdpe']:
            # np.load('discrete1/data/phi_{}_15.npy'.format(problem))
            problem = 'carbon'
        elif problem == 'pluto':
            print('Pluto')
            return np.load('discrete1/data/phi_group_{}.npy'.format(str(group).zfill(3)))
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
        from discrete1.setup import problem1,problem2
        # List of the material splits
        if problem == 'pluto':
            mat_split = problem2.boundaries_mat(distance=distance)
        else:
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

    def low_rank_svd_squeeze(big,small,rank,distance):
        import numpy as np
        from discrete1.util import sn
        from discrete1.setup import problem2,func

        phi_big = func.initial_flux('pluto',big)
        phi_small = func.initial_flux('pluto',small)
        scatter_big,fission_big = problem2.scatter_fission(big,distance)
        scatter_small,fission_small = problem2.scatter_fission(small,distance)
        # %%

        mat_split = problem2.boundaries_mat(distance)
        r_scatter = np.empty(scatter_small.shape)
        r_fission = np.empty(fission_small.shape)

        for mat in mat_split:
            # Big Phi
            phi_ = phi_big[mat].T.copy()
            u_p_big,_,_ = sn.svd(phi_,rank)
            smult_ = scatter_big[mat][0] @ phi_
            u_s_big,_,_ = sn.svd(smult_,rank)
            fmult_ = fission_big[mat][0] @ phi_
            u_f_big,_,_ = sn.svd(fmult_,rank)
            del phi_, fmult_, smult_
            # Creating scatter and fission tilde
            s_tilde = u_s_big.T @ scatter_big[mat][0] @ u_p_big
            f_tilde = u_f_big.T @ fission_big[mat][0] @ u_p_big
            
            # Small Phi
            phi_ = phi_small[mat].T.copy()
            u_p_small,_,_ = sn.svd(phi_,rank)
            smult_ = scatter_small[mat][0] @ phi_
            u_s_small,_,_ = sn.svd(smult_,rank)
            fmult_ = fission_small[mat][0] @ phi_
            u_f_small,_,_ = sn.svd(fmult_,rank)
            del phi_, fmult_, smult_
            # Creating scatter and fission tilde
            ys_tilde = u_s_small.T @ s_tilde @ u_p_small
            yf_tilde = u_f_small.T @ f_tilde @ u_p_small

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
        elif dtype in ['scatter']:
            print('Loading DJINN Scatter...')
            model_scatter = djinn.load(model_name=model_name)
            model_fission = None
        elif dtype in ['fission']:
            model_scatter = None
            model_fission = djinn.load(model_name=model_name)
        return model_scatter,model_fission

    def djinn_load_double(model_name,dtype):
        fuel_scatter,fuel_fission = func.djinn_load(model_name[0],dtype)
        refl_scatter,refl_fission = func.djinn_load(model_name[1],dtype)
        return fuel_scatter,fuel_fission,refl_scatter,refl_fission

    def auto_load(model_name,dtype,number=''):
        from tensorflow import keras
        if len(number) > 1:
            number = '_' + number
        print('Loading phi encoder...')
        phi_encoder = keras.models.load_model('{}_phi_encoder{}.h5'.format(model_name,number),compile=False)
        fmult_decoder = None; smult_decoder = None
        if dtype in ['fmult','fission','both']:
            print('Loading fmult decoder...')
            fmult_decoder = keras.models.load_model('{}_fmult_decoder{}.h5'.format(model_name,number),compile=False)    
        if dtype in ['smult','scatter','both']:
            print('Loading smult decoder...')
            smult_decoder = keras.models.load_model('{}_smult_decoder{}.h5'.format(model_name,number),compile=False)    
        return phi_encoder,fmult_decoder,smult_decoder

    def load_coder(coder,ptype='phi',number=''):
        """ Coder is the string path to the autoencoder, encoder, and decoder """
        from tensorflow import keras
        if len(number) > 1:
            number = '_' + number
        if coder == 'dummy1':
            return func.load_dummy1(),func.load_dummy1(),func.load_dummy1()
        elif coder == 'dummy2':
            return func.load_dummy2(),func.load_dummy2(),func.load_dummy2()
        elif coder == 'dummy3':
            return func.load_dummy3(),func.load_dummy3(),func.load_dummy3()
        print('Loading {} autoencoder...'.format(ptype))
        autoencoder = keras.models.load_model('{}_{}_autoencoder{}.h5'.format(coder,ptype,number))
        encoder = keras.models.load_model('{}_{}_encoder{}.h5'.format(coder,ptype,number),compile=False)
        decoder = keras.models.load_model('{}_{}_decoder{}.h5'.format(coder,ptype,number),compile=False)
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
    def variables(conc,problem=None,dim=618,distance=[5,1.5,3.5]): #
        import numpy as np
        from discrete1.util import sn
        I = 1000; 
        delta = sum(distance)/I
        # Scattering
        pu239_scatter = np.load('mydata/pu239/scatter_{}.npy'.format(str(dim).zfill(3)))
        pu240_scatter = np.load('mydata/pu240/scatter_{}.npy'.format(str(dim).zfill(3)))
        hdpe_scatter = np.load('mydata/hdpe/scatter_{}.npy'.format(str(dim).zfill(3)))

        enrich_scatter = pu239_scatter*(1-conc) + pu240_scatter*conc
        del pu239_scatter
        # Fission
        pu239_fission = np.load('mydata/pu239/nu_fission_{}.npy'.format(str(dim).zfill(3)))
        pu240_fission = np.load('mydata/pu240/nu_fission_{}.npy'.format(str(dim).zfill(3)))
        hdpe_fission = np.zeros((dim,dim))

        enrich_fission = pu239_fission*(1-conc) + pu240_fission*conc
        del pu239_fission
        # Total
        pu239_total = np.load('mydata/pu239/total_{}.npy'.format(str(dim).zfill(3)))
        pu240_total = np.load('mydata/pu240/total_{}.npy'.format(str(dim).zfill(3)))
        hdpe_total = np.load('mydata/hdpe/total_{}.npy'.format(str(dim).zfill(3)))

        enrich_total = pu239_total*(1-conc) + pu240_total*conc
        del pu239_total
        # Ordering
        xs_scatter = [hdpe_scatter,enrich_scatter,pu240_scatter]
        xs_total = [hdpe_total,enrich_total,pu240_total]
        xs_fission = [hdpe_fission,enrich_fission,pu240_fission]
        
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

    def boundaries_aux(conc,problem=None,distance=[5,1.5,3.5]):
        # Concentration is the amount of Pu-240 in Enriched
        import numpy as np
        from discrete1.util import sn
        # Fission Models
        if problem == 'pluto':
            ment = [(1-conc),1]; where = [1,2]
        # Scatter Models
        elif problem == 'pluto_full':
            ment = [15.04,(1-conc),1]; where = [0,1,2]

        I = 1000; delta = sum(distance)/I
        layers = [int(ii/delta) for ii in distance]
        # Use for layer correction
        if sum(layers) != sum(distance):
            change = I - sum(layers)
            if change % 2 != 0:
                layers[2] = layers[2] + 1
                change -= 1
            layers[0] = layers[0] + int(0.5*change)
            layers[-1] = layers[-1] + int(0.5*change)
        splits = np.array(sn.layer_slice(layers))
        # conc is uh3 enrich while 0 is depleted uranium
        enrichment = sn.enrich_list(sum(layers),ment,splits[where].tolist())
        return enrichment,sn.layer_slice_dict(layers,where)

    def boundaries(conc=0.15,problem=None,distance=[5,1.5,3.5]):
        # problem_scatter = problem + '_full'
        problem_scatter = problem
        # Set Fission Splits
        enrichment,splits = problem2.boundaries_aux(conc,problem,distance)
        fission_splits = {f'fission_{kk}': vv for kk, vv in splits.items()}
        # Set Scatter Splits
        enrichment,splits = problem2.boundaries_aux(conc,problem_scatter,distance)
        scatter_splits = {f'scatter_{kk}': vv for kk, vv in splits.items()}
        combo_splits = {**scatter_splits, **fission_splits}
        return enrichment,combo_splits


    def scatter_fission(enrich,dim=618,distance=[5,1.5,3.5]):
        _,_,_,_,_,scatter,fission,_,_,_ = problem2.variables(enrich,'pluto',dim,distance)
        return scatter,fission

    def scatter_fission_total(conc,dim=618):
        import numpy as np
        pu239_scatter = np.load('mydata/pu239/scatter_{}.npy'.format(str(dim).zfill(3)))
        pu240_scatter = np.load('mydata/pu240/scatter_{}.npy'.format(str(dim).zfill(3)))
        hdpe_scatter = np.load('mydata/hdpe/scatter_{}.npy'.format(str(dim).zfill(3)))

        enrich_scatter = pu239_scatter*conc + pu240_scatter*(1-conc)

        scatter = np.vstack((hdpe_scatter[None,:,:],enrich_scatter[None,:,:],pu240_scatter[None,:,:]))
        del pu239_scatter,pu240_scatter,hdpe_scatter,enrich_scatter
        # Fission
        pu239_fission = np.load('mydata/pu239/nu_fission_{}.npy'.format(str(dim).zfill(3)))
        pu240_fission = np.load('mydata/pu240/nu_fission_{}.npy'.format(str(dim).zfill(3)))
        hdpe_fission = np.zeros((dim,dim))

        enrich_fission = pu239_fission*(1-conc) + pu240_fission*conc

        fission = np.vstack((hdpe_fission[None,:,:],enrich_fission[None,:,:],pu240_fission[None,:,:]))
        del pu239_fission,pu240_fission,hdpe_fission,enrich_fission
        
        # Total
        pu239_total = np.load('mydata/pu239/total_{}.npy'.format(str(dim).zfill(3)))
        pu240_total = np.load('mydata/pu240/total_{}.npy'.format(str(dim).zfill(3)))
        hdpe_total = np.load('mydata/hdpe/total_{}.npy'.format(str(dim).zfill(3)))

        enrich_total = pu239_total*(1-conc) + pu240_total*conc

        total = np.vstack((hdpe_total[None,:],enrich_total[None,:],pu240_total[None,:]))
        del pu239_total,pu240_total,hdpe_total,enrich_total

        return scatter,fission,total

    def boundaries_mat(distance=[5,1.5,3.5]):
        import numpy as np
        from discrete1.util import sn
        I = 1000; delta = sum(distance)/I
        layers = [int(ii/delta) for ii in distance]
        # Use for layer correction
        if sum(layers) != sum(distance):
            change = I - sum(layers)
            if change % 2 != 0:
                layers[2] = layers[2] + 1
                change -= 1
            layers[0] = layers[0] + int(0.5*change)
            layers[-1] = layers[-1] + int(0.5*change)
        return np.sort(np.array(sn.layer_slice(layers)))


class problem1:        
    def variables(conc=None,problem=None,distance=[45,35,20],reduced=False):
        from discrete1.util import chem,sn
        import numpy as np
        if problem in ['stainless','carbon','hdpe']:
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
        if problem in ['carbon','hdpe']:
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

        if reduced:
            energy_grid = np.load('discrete1/data/energyGrid.npy')
            dim = reduced
            hdpe_total,hdpe_scatter,hdpe_fission = sn.group_reduction(dim,energy_grid,
                total=hdpe_total,scatter=hdpe_scatter.T,fission=hdpe_fission.T)
            uh3_total,uh3_scatter,uh3_fission = sn.group_reduction(dim,energy_grid,
                total=uh3_total,scatter=uh3_scatter.T,fission=uh3_fission.T)
            uh3_238_total,uh3_238_scatter,uh3_238_fission = sn.group_reduction(dim,energy_grid,
                total=uh3_238_total,scatter=uh3_238_scatter.T,fission=uh3_238_fission.T)
            # Don't need to be transposed
            xs_scatter = [hdpe_scatter,uh3_scatter,uh3_238_scatter]
            xs_total = [hdpe_total,uh3_total,uh3_238_total]
            xs_fission = [hdpe_fission,uh3_fission,uh3_238_fission]

        # Setting up eigenvalue equation
        N = 8; L = 0; R = sum(distance); #G = dim
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
        
        return dim,N,mu,w,total_,scatter_[:,0],fission_,L,R,I
    
    def boundaries_aux(conc,problem=None,distance=[45,35,20]):
        import numpy as np
        from discrete1.util import sn
        # Fission Models
        if problem in ['carbon','stainless']:
            # distance = [45,35,20]; 
            ment = [conc,0]; where = [1,2]
        elif problem ==  'mixed1':
            distance = [45,5,25,5,20]; ment = [0.12,0.27,0.12,0]; where = [1,2,3,4]
        elif problem == 'multiplastic':
            distance = [10]*8; distance.append(20); ment = [conc]*4; ment.append(0); where = [1,3,5,7,8]

        elif problem == 'hdpe':
            ment = [15.04]; where = [0]
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
        #problem_scatter = problem + '_full'
        problem_scatter = problem
        # Set Fission Splits
        enrichment,splits = problem1.boundaries_aux(conc,problem,distance)
        fission_splits = {f'fission_{kk}': vv for kk, vv in splits.items()}
        # Set Scatter Splits
        enrichment,splits = problem1.boundaries_aux(conc,problem_scatter,distance)
        scatter_splits = {f'scatter_{kk}': vv for kk, vv in splits.items()}
        combo_splits = {**scatter_splits, **fission_splits}
        return enrichment,combo_splits

    def scatter_fission(conc,problem,distance=[45,35,20],reduced=False):
        _,_,_,_,_,scatter,fission,_,_,_ = problem1.variables(conc,problem,distance,reduced)
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



