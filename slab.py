class func:
    def scattering_approx(L,mu):
        '''For Loop of the scattering for the transport equation
        Arguments:
            L: Number of moments
            mu: angle for which to use
        Returns:
            1 x L+1 vector of the scattering cross section
        '''
        from scipy.special import legendre as P
        import numpy as np
        scat = np.zeros(L+1)
        for l in range(L+1):
            scat[l] = (2*l+1)*P(l)(mu)
        return scat
    
    def total_add(tot,mu,delta,side):
        ''' Adding the scalars in the transport equation
        Arguments:
            tot: the total cross section at a given index
            mu: vector of angles between -1 and 1
            delta: the width of a spatial cell
            side: which side of the equation we are talking about
        Returns: 
            scalar of the mu/delta and total cross section/2 value
        '''
        if side == 'right' and mu > 0: #forward sweep
            return mu/delta-(0.5*tot)
        elif side == 'right' and mu < 0: #backward sweep
            return -mu/delta-(0.5*tot)
        elif side == 'left' and mu > 0: #forward sweep
            return mu/delta+(0.5*tot)
        return -mu/delta+(0.5*tot) #backward sweep
    
    def diamond_diff(top,bottom):
        return 0.5*(top + bottom)

    def update_q(N,L,mu,scatter,phi,start,stop,g):
        import numpy as np
        if L == 0:
            return np.tile(np.sum(scatter[:,:,g,start:stop]*phi[:,:,start:stop],axis=(2,1)),(N,1)).T
        return np.array([np.sum(func.scattering_approx(L,mu[n]).reshape(1,L+1,1)*scatter[:,:,g,start:stop]*phi[:,:,start:stop],axis=(2,1)) for n in range(N)]).T
        
    def temp_y_djinn(phi,scatter,djinn_y,bounds):
        import numpy as np
        ind = [slice(bounds[ii],bounds[ii+1]) for ii in range(len(bounds)-1)]
        aa = phi.copy(); bb = scatter.copy(); cc = djinn_y.copy()
        ops = ['bb[jj]*aa[jj]','np.expand_dims(cc,1)',
               'bb[jj]*aa[jj]','np.flip(np.expand_dims(cc,1),0)',
               'bb[jj]*aa[jj]']
        temp = []
        for jj,kk in zip(ind,ops):
            temp.append(eval(kk))
        del aa,bb,cc
        return np.concatenate(temp)
    #     return np.concatenate([eval(kk) for jj,kk in zip(ind,ops)])

    def update_q_djinn(N,mu,scatter,phi,start,stop,g,djinn_y,bounds):
        ''' This does not account for djinn with L > 0'''
        import numpy as np
        ind = [slice(bounds[ii],bounds[ii+1]) for ii in range(len(bounds)-1)]
        aa = phi.copy(); bb = scatter.copy(); cc = djinn_y.copy()
        ops = ['np.sum(bb[jj,:,g,start:stop]*aa[jj,:,start:stop],axis=(2,1))',
               'np.sum(cc[:,g,start:stop],axis=1)',
               'np.sum(bb[jj,:,g,start:stop]*aa[jj,:,start:stop],axis=(2,1))',
               'np.sum(np.flip(cc[:,g,start:stop],0),axis=1)',
               'np.sum(bb[jj,:,g,start:stop]*aa[jj,:,start:stop],axis=(2,1))']
        temp = []
        for jj,kk in zip(ind,ops):
            temp.append(eval(kk))
        del aa,bb,cc
        return np.tile(np.concatenate(temp),(N,1)).T
    #     return np.tile(np.concatenate([eval(kk) for jj,kk in zip(ind,ops)]),(N,1)).T
    
    def update_q_inf(N,L,mu,scatter,phi,start,stop,g):
        import numpy as np
        if L == 0:
            return np.tile(np.sum(scatter[:,g,start:stop]*phi[:,start:stop],axis=(1,0)),(N,1))
        return np.array([np.sum(func.scattering_approx(L,mu[n]).reshape(L+1,1)*scatter[:,g,start:stop]*phi[:,start:stop],axis=(1,0)) for n in range(N)])
    
    def update_q_inf_djinn(N,L,mu,djinn_y,start,stop,g):
        import numpy as np
        if L == 0:
            return np.tile(np.sum(djinn_y[:,g,start:stop],axis=1),(N,1))
        return np.array([np.sum(func.scattering_approx(L,mu[n]).reshape(L+1,1)*djinn_y[:,g,start:stop],axis=(1,0)) for n in range(N)])
# =============================================================================
#     def djinn_predict(name,data,mirror=False):
#         from djinn import djinn
#         model = djinn.load(model_name=name)
#         predicted = model.predict(data)
#         if mirror:
#             return func.reflect(predicted)
#         return predicted
#             
#     def reflect(data):
#         import numpy as np
#         mirror = np.empty(tuple(list([data.shape[0]*2]+[data.shape[1]])))
#         mirror[:data.shape[0]] = data.copy()
#         mirror[data.shape[0]:] = np.flip(data.copy(),0)
#         return mirror
# =============================================================================
    

class eigen:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L,R,I):
        self.G = G
        self.N = N
        self.mu = mu
        self.w = w
        self.total = total
        self.scatter = scatter
        self.chiNuFission = chiNuFission
        self.L = L
        self.I = I
        self.delta = float(R)/I
        
    def one_group(N,mu,w,total,scatter,L,external,I,delta,guess,tol=1e-08,MAX_ITS=100,LOUD=False):
        ''' 
        Arguments:
            N: Number of angles
            mu: vector of angles between -1 and 1
            w: vector of weights for the angles
            total: I x 1 vector of the total cross section for each spatial cell
            scatter: I x L+1 array for the scattering of the spatial cell by moment
            L: Number of moments
            external: I x N array for the external sources
            I: Number of spatial cells
            delta: width of one spatial cell
            guess: Initial guess of the scalar flux for a specific energy group (I x L+1)
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iterations and change between iterations, default is False
        Returns:
            phi: a I x L+1 array
        '''
        import numpy as np
        
        converged = 0
        count = 1
        
        phi = np.zeros((I,L+1))
        phi_old = guess.copy()
        while not(converged):
            phi *= 0
            for n in range(N):
                non_weight_scatter = func.scattering_approx(L,mu[n])
                weight_scatter = w[n]*non_weight_scatter
                temp_scat = np.sum(non_weight_scatter*scatter*phi_old,axis=1)
                if  mu[n] > 0:
                    for ii in range(I):
                        if ii == 0:
                            psi_bottom = 0 #vacuum
                        else:
                            psi_bottom = psi_top
                        psi_top = (temp_scat[ii] + external[ii,n] + psi_bottom * func.total_add(total[ii], mu[n], delta,'right'))/(func.total_add(total[ii], mu[n], delta,'left'))
                        phi[ii,:] = phi[ii,:] + (weight_scatter * func.diamond_diff(psi_top,psi_bottom))
                elif mu[n] < 0:
                    for ii in range(I-1,-1,-1):
                        if ii == I - 1:
                            psi_top = 0 #vacuum
                        else:
                            psi_top = psi_bottom
                        psi_bottom = (temp_scat[ii] + external[ii,n] + psi_top * func.total_add(total[ii], mu[n], delta,'right'))/(func.total_add(total[ii], mu[n], delta,'left'))
                        phi[ii,:] = phi[ii,:] +  (weight_scatter * func.diamond_diff(psi_top,psi_bottom)).flatten()
            change = np.linalg.norm((phi - phi_old)/phi/(I*(L+1)))
            converged = (change < tol) or (count >= MAX_ITS) 
            if LOUD:
                print('Iteration:',count,' Change:',change)
            count += 1
            phi_old = phi.copy()

        return phi
    
    def multi_group(G,N,mu,w,total,scatter,L,nuChiFission,I,delta,tol=1e-08,MAX_ITS=100,LOUD=False):
        '''
        Arguments:
            G: number of energy groups
            N: number of angles
            mu: vector of angles between -1 and 1
            w: vector of weights for the angles
            total: I x G vector of the total cross section for each spatial cell and energy level
            scatter: I x L+1 x G array for the scattering of the spatial cell by moment and energy
            L: Number of moments
            nuChiFission: 
            I: Number of spatial cells
            delta: width of one spatial cell
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a I x L+1 x G array
        '''
        import numpy as np
        phi_old = np.zeros((I,L+1,G))
        converged = 0
        count = 1
        while not (converged):
            if LOUD:
                print('New Inner Loop Iteration\n======================================')
            phi = np.zeros(phi_old.shape)
            for g in range(G):
                if (LOUD):
                    print("Inner Transport Iterations\n===================================")
                if g == 0:
                    q_tilde = nuChiFission[:,g].reshape(I,1) + func.update_q(N,L,mu,scatter,phi_old,g+1,G,g)
                else:
                    q_tilde = nuChiFission[:,g].reshape(I,1) + func.update_q(N,L,mu,scatter,phi_old,g+1,G,g) + func.update_q(N,L,mu,scatter,phi,0,g,g)
                phi[:,:,g] = eigen.one_group(N,mu,w,total[:,g],scatter[:,:,g,g],L,q_tilde,I,delta,tol=tol,MAX_ITS=MAX_ITS,LOUD=LOUD,guess=phi_old[:,:,g])
            change = np.linalg.norm((phi - phi_old)/phi/(I*(L+1)))
            if LOUD:
                print('Change is',change,count)
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi

    def transport(self,tol=1e-12,MAX_ITS=100,LOUD=True,dimen=False):
        ''' 
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
            phi: a I x L+1 x G array
        '''
        import numpy as np
        from discrete1.util import sn_tools

        phi_old = np.random.rand(self.I,self.L+1,self.G)
        k_old = np.linalg.norm(phi_old)
        phi_old /= k_old
        converged = 0
        count = 1
        # Dimensionality Purposes
        if dimen:
            scatterSn = sn_tools.propagate(xs=self.scatter,I=self.I,L=self.L,G=self.G,dtype='scatter')
            totalSn = sn_tools.propagate(xs=self.total,I=self.I,G=self.G,dtype='total')
        else:
            scatterSn = self.scatter.copy()
            totalSn = self.total.copy()
            
        sources = phi_old[:,0,:] * self.chiNuFission #np.sum(self.chiNuFission,axis=0)

        while not (converged):
            if LOUD:
                print('Outer Transport Iteration\n===================================')
            phi = eigen.multi_group(self.G,self.N,self.mu,self.w,totalSn,scatterSn,self.L,sources,self.I,self.delta,tol=1e-08,MAX_ITS=MAX_ITS,LOUD=False)

            keff = np.linalg.norm(phi)
            phi /= np.linalg.norm(phi)
            k_change = np.fabs(keff - k_old)
            change = np.linalg.norm((phi-phi_old)/phi/(self.I*(self.L+1)))
            if LOUD:
                print('Change is',change,count,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS)
            count += 1
            k_old = keff
            phi_old = phi.copy()
            sources = phi_old[:,0,:] * self.chiNuFission # np.sum(self.chiNuFission,axis=0)
        return phi,keff

    
class eigen_djinn:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L,R,I):
        self.G = G
        self.N = N
        self.mu = mu
        self.w = w
        self.total = total
        self.scatter = scatter
        self.chiNuFission = chiNuFission
        self.L = L
        self.I = I
        self.delta = float(R)/I
    
    def one_group(N,mu,w,total,scatter,L,external,I,delta,guess,djinn_1g,ezone,tol=1e-08,MAX_ITS=100,LOUD=False):
        ''' DJINN One Energy Group Calculation
        Arguments:
            N: Number of angles
            mu: list of angles between -1 and 1
            w: list of weights for the angles
            total: I x 1 vector of the total cross section for each spatial cell
            scatter: I x L+1 array for the scattering of the spatial cell by moment
            L: Number of moments
            external: I x N array for the external sources
            I: Number of spatial cells
            delta: width of one spatial cell
            guess: Initial guess of the scalar flux for a specific energy group (I x L+1)
            djinn_1g: I x 1 list of DJINN predicted sigma_s*phi 
            ezones: list of boundary cells between compounds
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iterations and change between iterations, default is False
        Returns:
            phi: a I x L+1 array '''
        import numpy as np
        # Initialize phi_old and phi
        phi = np.zeros((I,L+1))
        phi_old = guess.copy()
        # sigma_s*phi multiplication for initialized phi_old
        temp_y = func.temp_y_djinn(phi_old,scatter,djinn_1g,ezone)
        # Parameters for convergence/count
        converged = 0
        count = 1
        while not(converged):
            # zero out new phi matrix
            phi *= 0
            # Iterate over all the observable angles
            for n in range(N):
                # Calculate (2l+1)P(l)(mu)
                non_weight_scatter = func.scattering_approx(L,mu[n])
                # Multiply by weight w(n)
                weight_scatter = w[n]*non_weight_scatter
                # Combine (2l+1)P(l)(mu) and sigma_s*phi for the scattering term
                temp_scat = np.sum(non_weight_scatter*temp_y,axis=1)
                if  mu[n] > 0: # left to right sweep
                    for ii in range(I): # Iterate over spatial cells
                        if ii == 0:
                            psi_bottom = 0 # vacuum boundary for eigenvalue
                        else:
                            psi_bottom = psi_top # use angular flux from previous cell edge
                        # Calculate angular flux at other cell edge
                        psi_top = (temp_scat[ii] + external[ii,n] + psi_bottom * func.total_add(total[ii], mu[n], delta,'right'))/(func.total_add(total[ii], mu[n], delta,'left')) 
                        # Update phi for spatial cell center
                        phi[ii,:] = phi[ii,:] + (weight_scatter * func.diamond_diff(psi_top,psi_bottom))
                elif mu[n] < 0: # Right to left sweep
                    for ii in range(I-1,-1,-1): # Iterate cells backwards
                        if ii == I - 1:
                            psi_top = 0 # vacuum boundary for eigenvalue
                        else:
                            psi_top = psi_bottom # use angular flux from previous cell edge
                        # Calculate angular flux at other cell edge
                        psi_bottom = (temp_scat[ii] + external[ii,n] + psi_top * func.total_add(total[ii], mu[n], delta,'right'))/(func.total_add(total[ii], mu[n], delta,'left'))
                        # Update phi for spatial cell center
                        phi[ii,:] = phi[ii,:] +  (weight_scatter * func.diamond_diff(psi_top,psi_bottom)).flatten()
            # Check for convergence
            change = np.linalg.norm((phi - phi_old)/phi/(I*(L+1)))
            converged = (change < tol) or (count >= MAX_ITS) 
            if LOUD:
                print('Iteration:',count,' Change:',change)
            count += 1
            # Update phi
            phi_old = phi.copy()
            # Update sigma_s * phi
            temp_y = func.temp_y_djinn(phi_old,scatter,djinn_1g,ezone) # I did not have this before
        return phi

    def multi_group(G,N,mu,w,total,scatter,L,nuChiFission,I,delta,djinn_mg,zones,tol=1e-08,MAX_ITS=100,LOUD=False):
        ''' DJINN Multigroup Calculation (Source Iteration)
        Arguments:
            G: number of energy groups
            N: number of angles
            mu: list of angles between -1 and 1
            w: vector of weights for the angles
            total: I x G array of the total cross section 
            scatter: I x L+1 x G array for the scattering 
            L: Number of scattering moments
            nuChiFission: I x G external source
            I: Number of spatial cells
            delta: width of one spatial cell
            djinn_mg: I x G x G array of DJINN predicted sigma_s*phi 
            zones: list of boundary cells between compounds
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a I x L+1 x G array  '''
        import numpy as np
        # Initialize phi
        phi_old = np.zeros((I,L+1,G))
        # Parameters for convergence/count
        converged = 0
        count = 1
        while not (converged):
            if LOUD:
                print('New Inner Loop Iteration\n======================================')
            # zero out new phi
            phi = np.zeros(phi_old.shape)
            # Iterate over all the energy groups
            for g in range(G):
                if (LOUD):
                    print("Inner Transport Iterations\n===================================")
                # Calculate the source term (takes downscattering matrix with update_q_djinn function)
                if g == 0:
                    q_tilde = nuChiFission[:,g].reshape(I,1) + func.update_q_djinn(N,mu,scatter,phi_old,g+1,G,g,djinn_mg,zones)
                # Calculate the source term (downscattering and upscattering are separated)
                else:
                    q_tilde = nuChiFission[:,g].reshape(I,1) + func.update_q_djinn(N,mu,scatter,phi_old,g+1,G,g,djinn_mg,zones) + func.update_q_djinn(N,mu,scatter,phi,0,g,g,djinn_mg,zones)
                # Assigns new value to phi I x L+1
                phi[:,:,g] = eigen_djinn.one_group(N,mu,w,total[:,g],scatter[:,:,g,g],L,q_tilde,I,delta,phi_old[:,:,g],djinn_mg[:,g,g],zones,tol=tol,MAX_ITS=MAX_ITS,LOUD=LOUD)
            # Check for convergence
            change = np.linalg.norm((phi - phi_old)/phi/(I*(L+1)))
            if LOUD:
                print('Change is',change,count)
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            # Update phi and repeat
            phi_old = phi.copy()
        return phi
    
    def transport(self,model_name,zones,tol=1e-12,MAX_ITS=100,LOUD=True,dimen=False):
        ''' DJINN Multigroup Calculation (Power Iteration)
        Arguments:
            model_name: file location of DJINN model
            zones: list of cell locations where compounds change
            tol: tolerance of convergence, default is 1e-12
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
            dimen: For dimensionality purposes with scattering and total cross section - should remove?
        Returns:
            phi: a I x L+1 x G array  '''
        import numpy as np
        from discrete1.util import sn_tools
        from djinn import djinn
        #Initialize and normalize phi
        phi_old = np.random.rand(self.I,self.L+1,self.G)
        k_old = np.linalg.norm(phi_old)
        phi_old /= k_old
        # Load DJINN Model
        model = djinn.load(model_name=model_name)
        # Set width of enriched zone 
        enriched_width = zones[2] - zones[1] 
        # Predict enriched zone y with normalized phi
        djinn_y = model.predict(phi_old[zones[1]:zones[2],0,:]).reshape(enriched_width,self.G,self.G)
        # Dimensionality Purposes
        if dimen:
            scatterSn = sn_tools.propagate(xs=self.scatter,I=self.I,L=self.L,G=self.G,dtype='scatter')
            totalSn = sn_tools.propagate(xs=self.total,I=self.I,G=self.G,dtype='total')
        else:
            scatterSn = self.scatter.copy()
            totalSn = self.total.copy()
        # Set sources for power iteration    
        sources = phi_old[:,0,:] * self.chiNuFission 
        # Parameters for convergence/count 
        converged = 0
        count = 1
        while not (converged):
            if LOUD:
                print('Outer Transport Iteration\n===================================')
            # Calculate phi with original sources
            phi = eigen_djinn.multi_group(self.G,self.N,self.mu,self.w,totalSn,scatterSn,self.L,sources,self.I,self.delta,djinn_y,zones,tol=1e-08,MAX_ITS=MAX_ITS,LOUD=False)
            # Calculate k-effective
            keff = np.linalg.norm(phi)
            # Normalize to 1
            phi /= np.linalg.norm(phi)
            # Check for convergence
            #k_change = np.fabs(keff - k_old)
            change = np.linalg.norm((phi-phi_old)/phi/(self.I*(self.L+1)))
            if LOUD:
                print('Change is',change,count,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS) or (k_change < tol)
            count += 1
            #Update phi,keff
            k_old = keff
            phi_old = phi.copy()
            # Update y
            djinn_y = model.predict(phi_old[zones[1]:zones[2],0,:]).reshape(enriched_width,self.G,self.G)
            # Update sources
            sources = phi_old[:,0,:] * self.chiNuFission 
        return phi,keff

class inf_eigen:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L):
        self.G = G
        self.N = N
        self.mu = mu
        self.w = w
        self.total = total
        self.scatter = scatter
        self.chiNuFission = chiNuFission
        self.L = L
   
    def one_group(N,mu,w,total,scatter,L,external,guess,tol=1e-08,MAX_ITS=100,LOUD=False):   
        ''' 
        Arguments:
            N: Number of angles
            mu: vector of angles between -1 and 1
            w: vector of weights for the angles
            total: I x 1 vector of the total cross section for each spatial cell
            scatter: I x L+1 array for the scattering of the spatial cell by moment
            L: Number of moments
            external: I x N array for the external sources
            I: Number of spatial cells
            delta: width of one spatial cell
            guess: Initial guess of the scalar flux for a specific energy group (I x L+1)
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iterations and change between iterations, default is False
        Returns:
            phi: a I x L+1 array
        '''
        import numpy as np
        
        converged = 0
        count = 1
        phi = np.zeros((L+1))
        phi_old = guess.copy()
        
        while not(converged):
            phi *= 0
            for n in range(N):
                non_weight_scatter = func.scattering_approx(L,mu[n])
                weight_scatter = w[n]*non_weight_scatter
                temp_scat = np.sum(non_weight_scatter*scatter*phi_old,axis=0)
                psi = (temp_scat + external[n])/(total)
                phi = phi + w[n] * non_weight_scatter * psi
            change = np.linalg.norm((phi - phi_old)/phi/((L+1)))
            converged = (change < tol) or (count >= MAX_ITS) 
            if LOUD:
                print('Iteration:',count,' Change:',change)
            count += 1
            phi_old = phi.copy()
        return phi


    def multi_group(G,N,mu,w,total,scatter,L,nuChiFission,tol=1e-08,MAX_ITS=100,LOUD=False):
        '''
        Arguments:
            G: number of energy groups
            N: number of angles
            mu: vector of angles between -1 and 1
            w: vector of weights for the angles
            total: G x 1 vector of the total cross section for each spatial cell and energy level
            scatter: L+1 x G array for the scattering of the spatial cell by moment and energy
            L: Number of moments
            nuChiFission: 
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a L+1 x G array
        '''
        import numpy as np
        phi_old = np.zeros((L+1,G))
        converged = 0
        count = 1
        while not (converged):
            if LOUD:
                print('New Inner Loop Iteration\n======================================')
            phi = np.zeros(phi_old.shape)
            for g in range(G):
                if (LOUD):
                    print("Inner Transport Iterations\n===================================")
                if g == 0:
                    q_tilde = nuChiFission[g] + func.update_q_inf(N,L,mu,scatter,phi_old,g+1,G,g)
                else:
                    q_tilde = nuChiFission[g] + func.update_q_inf(N,L,mu,scatter,phi_old,g+1,G,g) + func.update_q_inf(N,L,mu,scatter,phi,0,g,g)
                phi[:,g] = inf_eigen.one_group(N,mu,w,total[g],scatter[:,g,g],L,q_tilde,phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS,LOUD=LOUD)
            change = np.linalg.norm((phi - phi_old)/phi/((L+1)))
            if LOUD:
                print('Change is',change,count)
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi

    def transport(self,tol=1e-12,MAX_ITS=100,LOUD=True):
        import numpy as np
        
        phi_old = np.random.rand(self.L+1,self.G)
        k_old = np.linalg.norm(phi_old)
        phi_old /= k_old
        
        sources = phi_old[0,:] * self.chiNuFission 
        
        converged = 0
        count = 1
        while not (converged):
            if LOUD:
                print('Outer Transport Iteration\n===================================')
            phi = inf_eigen.multi_group(self.G,self.N,self.mu,self.w,self.total,self.scatter.reshape(self.L+1,self.G,self.G),self.L,sources,tol=1e-08,MAX_ITS=MAX_ITS,LOUD=False)
            
            keff = np.linalg.norm(phi)
            phi /= np.linalg.norm(phi)
            k_change = np.fabs(keff - k_old)
            change = np.linalg.norm((phi-phi_old)/phi/((self.L+1)))
            if LOUD:
                print('Change is',change,count,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS)
            count += 1
            k_old = keff
            phi_old = phi.copy()
            sources = phi_old[0,:] * self.chiNuFission 

        return phi,keff    


class inf_eigen_djinn:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L):
        self.G = G
        self.N = N
        self.mu = mu
        self.w = w
        self.total = total
        self.scatter = scatter
        self.chiNuFission = chiNuFission
        self.L = L
   
    def one_group(N,mu,w,total,scatter,L,external,guess,djinn_1g,tol=1e-08,MAX_ITS=100,LOUD=False):   
        ''' Infinite DJINN One Group Calculation
        Arguments:
            N: Number of angles
            mu: vector of angles between -1 and 1
            w: vector of weights for the angles
            total: value of the total cross section for each spatial cell
            scatter: L+1 list for the scattering of the spatial cell by moment
            L: Number of moments
            external: 2 x N array for the external sources
            guess: Initial guess of the scalar flux for a specific energy group (L+1)
            djinn_1g: L+1 list of DJINN predicted sigma_s*phi
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iterations and change between iterations, default is False
        Returns:
            phi: a L+1 array
        '''
        import numpy as np
        # Initialize phi_old and phi
        phi = np.zeros((L+1))
        phi_old = guess.copy()
        # signa_s*phi multiplication for initialized phi_old
        
        # Parameters for convergence/count
        converged = 0
        count = 1
        while not(converged):
            # zero out new phi matrix
            phi *= 0
            # Iterate over all the observable angles
            for n in range(N):
                # Calculate (2l+1)P(l)(mu)
                non_weight_scatter = func.scattering_approx(L,mu[n])
# =============================================================================
#                 # Multiply by weight w(n)
#                 weight_scatter = w[n]*non_weight_scatter
# =============================================================================
                # Combine (2l+1)P(l)(mu) and sigma_s*phi for scattering term
                temp_scat = np.sum(non_weight_scatter*djinn_1g,axis=0)
                # Calculate angular flux
                psi = (temp_scat + external[n])/(total)
                # Update scalar flux
                phi = phi + w[n] * non_weight_scatter * psi
            # Check for convergence
            change = np.linalg.norm((phi - phi_old)/phi/((L+1)))
            converged = (change < tol) or (count >= MAX_ITS) 
            if LOUD:
                print('Iteration:',count,' Change:',change)
            count += 1
            # Update phi
            phi_old = phi.copy()
        return phi


    def multi_group(G,N,mu,w,total,scatter,L,nuChiFission,model_name,djinn_prediction,tol=1e-08,MAX_ITS=100,LOUD=False):
        ''' Infinite DJINN Multigroup Calculation (Source Iteration)
        Arguments:
            G: number of energy groups
            N: number of angles
            mu: vector of angles between -1 and 1
            w: vector of weights for the angles
            total: G x 1 vector of the total cross section for each spatial cell and energy level
            scatter: L+1 x G array for the scattering of the spatial cell by moment and energy
            L: Number of moments
            nuChiFission: G x 1 external source
            model_name: DJINN model
            djinn_prediction: Outer Iteration Predicted sigma_s*phi
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a L+1 x G array
        '''
        import numpy as np
        from djinn import djinn
        # Initialize phi
        phi_old = np.zeros((L+1,G))
        # Load DJINN model
        #djinn_model = djinn.load(model_name=model_name)
        # Parameters for convergence/count
        converged = 0
        count = 1
        while not (converged):
            if LOUD:
                print('New Inner Loop Iteration\n======================================')
            # zero out new phi
            phi = np.zeros(phi_old.shape)
            # Iterate over all the energy groups
            if count == 1:
                dj_pred = djinn_prediction.copy()
            else:
                dj_pred = djinn_prediction.copy()
                #dj_pred = djinn_model.predict(phi_old).reshape(L+1,G,G)
            for g in range(G):
                if (LOUD):
                    print("Inner Transport Iterations\n===================================")
                # Calculate the source term (takes downscattering matrix with djinn_mg)
                if g == 0:
                    q_tilde = nuChiFission[g] + func.update_q_inf_djinn(N,L,mu,dj_pred,g+1,G,g)
                # Calculate the source term (downscattering and upscattering are separated)
                else:
                    q_tilde = nuChiFission[g] + func.update_q_inf_djinn(N,L,mu,dj_pred,g+1,G,g) + func.update_q_inf_djinn(N,L,mu,dj_pred,0,g,g)
                # Assigns new value to phi L+1 
                phi[:,g] = inf_eigen_djinn.one_group(N,mu,w,total[g],scatter[:,g,g],L,q_tilde,phi_old[:,g],dj_pred[:,g,g],tol=tol,MAX_ITS=MAX_ITS,LOUD=LOUD)
            # Check for convergence
            #print('count',count,'sum',np.sum(phi))
            change = np.linalg.norm((phi - phi_old)/phi/((L+1)))
            if LOUD:
                print('Change is',change,count)
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            # Update phi and repeat
            phi_old = phi.copy()
        return phi

    def transport(self,model_name,tol=1e-12,MAX_ITS=100,LOUD=True):
        ''' Infinite DJINN Multigroup Calculation (Power Iteration) 
        Arguments:
            model_name: file location of DJINN model
            tol: tolerance of convergence, default is 1e-12
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a L+1 x G array  
            '''
        import numpy as np
        from djinn import djinn
# =============================================================================
#         import warnings
#         warnings.simplefilter('error', RuntimeWarning)
# =============================================================================
        # Initialize and normalize phi
        phi_old = np.random.rand(self.L+1,self.G)
        phi_old /= np.linalg.norm(phi_old)
        # Load DJINN model
        model = djinn.load(model_name=model_name)
        # Predict sigma_s*phi from initialized phi
        djinn_y = model.predict(phi_old).reshape(self.L+1,self.G,self.G)
        # Set sources for power iteration
        sources = phi_old[0,:] * self.chiNuFission 
        # Parameters for convergence/count
        converged = 0
        count = 1
        while not (converged):
            if LOUD:
                print('Outer Transport Iteration\n===================================')
            # Calculate phi with original sources
            phi = inf_eigen_djinn.multi_group(self.G,self.N,self.mu,self.w,self.total,self.scatter.reshape(self.L+1,self.G,self.G),self.L,sources,model_name,djinn_y,tol=1e-08,MAX_ITS=MAX_ITS,LOUD=False)
            # Calculate k-effective
            keff = np.linalg.norm(phi)
            # Normalize to 1
            phi /= np.linalg.norm(phi)
            # Check for convergence
            change = np.linalg.norm((phi-phi_old)/phi/((self.L+1)))
            if LOUD:
                print('Change is',change,count,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS)
            count += 1
            # Update phi
            phi_old = phi.copy()
            # Update djinn_y
            djinn_y = model.predict(phi_old).reshape(self.L+1,self.G,self.G)
            # Update sources
            sources = phi_old[0,:] * self.chiNuFission 
        return phi,keff    
    