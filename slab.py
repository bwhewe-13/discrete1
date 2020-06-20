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

class eigen:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L,R,I,track=False,enrich=None,splits=None):
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
        self.track = track
        self.enrich = enrich
        self.splits = splits
        
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
    
    def multi_group(self,G,N,mu,w,total,scatter,L,nuChiFission,I,delta,tol=1e-08,MAX_ITS=100,LOUD=False):
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
        # if self.track:            
        #     mymat = np.zeros((1,3,G))
        while not (converged):
            if LOUD:
                print('New Inner Loop Iteration\n======================================')
            phi = np.zeros(phi_old.shape)
            # if self.track == 'scatter' or self.track == 'both':
            #     for ii in self.splits:
            #         length = len(range(*ii.indices(I)))
            #         enrichment = np.repeat(self.enrich[ii],G).reshape(length,1,G)
            #         multiplier = np.einsum('ijk,ik->ij',scatter[ii][:,0],phi_old[ii][:,0]) # matrix multiplication
            #         track_temp = np.hstack((enrichment,phi_old[ii],multiplier.reshape(length,1,G)))
            #         mymat = np.vstack((mymat,track_temp))
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
        # if self.track:
        #     return phi,mymat[1:]
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
        from discrete1.util import sn

        phi_old = np.random.rand(self.I,self.L+1,self.G)
        phi_old /= np.linalg.norm(phi_old)
        converged = 0
        count = 1
        # Dimensionality Purposes
        if dimen:
            scatterSn = sn.propagate(xs=self.scatter,I=self.I,L=self.L,G=self.G,dtype='scatter')
            totalSn = sn.propagate(xs=self.total,I=self.I,G=self.G,dtype='total')
        else:
            scatterSn = self.scatter.copy()
            totalSn = self.total.copy()            
        sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old[:,0,:]) 
        if self.track:
            if self.track == 'scatter' or self.track == 'both':
                allmat_sca = np.zeros((1,3,self.G))
            if self.track == 'fission' or self.track == 'both':
                allmat_fis = np.zeros((1,3,self.G))
                for ii in self.splits:
                    length = len(range(*ii.indices(self.I)))
                    # Scattering
                    enrichment = np.repeat(self.enrich[ii],self.G).reshape(length,1,self.G)
                    multiplier = np.einsum('ijk,ik->ij',self.scatter[ii][:,0],phi_old[ii][:,0]) # matrix multiplication
                    track_temp = np.hstack((enrichment,phi_old[ii],multiplier.reshape(length,1,self.G)))
                    allmat_sca = np.vstack((allmat_sca,track_temp))
                    # Fission
                    enrichment = np.repeat(self.enrich[ii],self.G).reshape(length,1,self.G)
                    multiplier = np.einsum('ijk,ik->ij',self.chiNuFission[ii],phi_old[ii][:,0]) # matrix multiplication
                    track_temp = np.hstack((enrichment,phi_old[ii],multiplier.reshape(length,1,self.G)))
                    allmat_fis = np.vstack((allmat_fis,track_temp))
        while not (converged):
            if LOUD:
                print('Outer Transport Iteration\n===================================')
            # if self.track == 'scatter' or self.track == 'both':
            #     phi,tempmat = eigen.multi_group(self,self.G,self.N,self.mu,self.w,totalSn,scatterSn,self.L,sources,self.I,self.delta,tol=1e-08,MAX_ITS=MAX_ITS,LOUD=False)
            #     allmat_sca = np.vstack((allmat_sca,tempmat))
            # else:
            phi = eigen.multi_group(self,self.G,self.N,self.mu,self.w,totalSn,scatterSn,self.L,sources,self.I,self.delta,tol=1e-08,MAX_ITS=MAX_ITS,LOUD=False)

            keff = np.linalg.norm(phi)
            phi /= np.linalg.norm(phi)

            change = np.linalg.norm((phi-phi_old)/phi/(self.I*(self.L+1)))
            if LOUD:
                print('Change is',change,count,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS)
            count += 1
            phi_old = phi.copy()
            sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old[:,0,:]) 
            if self.track == 'fission' or self.track == 'both':
                for ii in self.splits:
                    length = len(range(*ii.indices(self.I)))
                    enrichment = np.repeat(self.enrich[ii],self.G).reshape(length,1,self.G)
                    multiplier = np.einsum('ijk,ik->ij',self.scatter[ii][:,0],phi_old[ii][:,0]) # matrix multiplication
                    track_temp = np.hstack((enrichment,phi_old[ii],multiplier.reshape(length,1,self.G)))
                    allmat_sca = np.vstack((allmat_sca,track_temp))
                    
                    enrichment = np.repeat(self.enrich[ii],self.G).reshape(length,1,self.G)
                    multiplier = np.einsum('ijk,ik->ij',self.chiNuFission[ii],phi_old[ii][:,0]) # matrix multiplication
                    track_temp = np.hstack((enrichment,phi_old[ii],multiplier.reshape(length,1,self.G)))
                    allmat_fis = np.vstack((allmat_fis,track_temp))
            # sources = phi_old[:,0,:] * self.chiNuFission # np.sum(self.chiNuFission,axis=0)
        if self.track:
            if self.track == 'both':
                return phi,keff,allmat_fis[1:],allmat_sca[1:]
            if self.track == 'scatter':
                return phi,keff,allmat_sca[1:]
            if self.track == 'fission':
                return phi,keff,allmat_fis[1:]
        return phi,keff

class eigen_symm:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L,R,I,track=False,enrich=None,splits=None):
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
        self.track = track
        self.enrich = enrich
        self.splits = splits
        
    def one_group(N,mu,w,total,scatter,L,external,I,delta,guess,tol=1e-08,MAX_ITS=100,LOUD=False):
        """ Arguments:
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
            phi: a I x L+1 array   """
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
                psi_bottom = 0 # vacuum on LHS
                # Left to right
                for ii in range(I):
                    psi_top = (temp_scat[ii] + external[ii,n] + psi_bottom * func.total_add(total[ii], mu[n], delta,'right'))/(func.total_add(total[ii], mu[n], delta,'left'))
                    phi[ii,:] = phi[ii,:] + (weight_scatter * func.diamond_diff(psi_top,psi_bottom))
                    psi_bottom = psi_top
                # Reflective right to left
                for ii in range(I-1,-1,-1):
                    psi_top = psi_bottom
                    psi_bottom = (temp_scat[ii] + external[ii,n] + psi_top * func.total_add(total[ii], -mu[n], delta,'right'))/(func.total_add(total[ii], -mu[n], delta,'left'))
                    phi[ii,:] = phi[ii,:] +  (weight_scatter * func.diamond_diff(psi_top,psi_bottom)).flatten()
            change = np.linalg.norm((phi - phi_old)/phi/(I*(L+1)))
            converged = (change < tol) or (count >= MAX_ITS) 
            if LOUD:
                print('Iteration:',count,' Change:',change)
            count += 1
            phi_old = phi.copy()
        return phi
    
    def multi_group(self,G,N,mu,w,total,scatter,L,nuChiFission,I,delta,tol=1e-08,MAX_ITS=100,LOUD=False):
        """ Arguments:
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
            phi: a I x L+1 x G array  """
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
                phi[:,:,g] = eigen_symm.one_group(N,mu,w,total[:,g],scatter[:,:,g,g],L,q_tilde,I,delta,tol=tol,MAX_ITS=MAX_ITS,LOUD=LOUD,guess=phi_old[:,:,g])
            change = np.linalg.norm((phi - phi_old)/phi/(I*(L+1)))
            if LOUD:
                print('Change is',change,count)
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi

    def transport(self,tol=1e-12,MAX_ITS=100,LOUD=True):
        """ Arguments:
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
        from discrete1.util import sn
        phi_old = np.random.rand(self.I,self.L+1,self.G)
        k_old = np.linalg.norm(phi_old)
        phi_old /= np.linalg.norm(phi_old)
        converged = 0
        count = 1
        if self.track:
            allmat_sca = np.zeros((1,3,self.G))
            allmat_fis = np.zeros((1,3,self.G))
        sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old[:,0,:]) 
        while not (converged):
            if self.track == 'scatter' or self.track == 'both':
                enrichment = np.expand_dims(np.tile(np.expand_dims(sn.cat(self.enrich,self.splits),axis=1),(1,self.G)),axis=1)
                multiplier = np.expand_dims(np.einsum('ijk,ik->ij',sn.cat(self.scatter,self.splits)[:,0],sn.cat(phi_old,self.splits)[:,0]),axis=1)
                track_temp = np.hstack((enrichment,sn.cat(phi_old,self.splits),multiplier))
                allmat_sca = np.vstack((allmat_sca,track_temp))
                # print(allmat_sca.shape)
            if self.track == 'fission' or self.track == 'both':
                enrichment = np.expand_dims(np.tile(np.expand_dims(sn.cat(self.enrich,self.splits),axis=1),(1,self.G)),axis=1)
                multiplier = np.expand_dims(np.einsum('ijk,ik->ij',sn.cat(self.chiNuFission,self.splits),sn.cat(phi_old,self.splits)[:,0]),axis=1)
                track_temp = np.hstack((enrichment,sn.cat(phi_old,self.splits),multiplier))
                allmat_fis = np.vstack((allmat_fis,track_temp))
                # print(allmat_fis.shape)
            if LOUD:
                print('Outer Transport Iteration {}\n==================================='.format(count))
            phi = eigen_symm.multi_group(self,self.G,self.N,self.mu,self.w,self.total,self.scatter,self.L,sources,self.I,self.delta,tol=1e-08,MAX_ITS=MAX_ITS,LOUD=False)
            keff = np.linalg.norm(phi)
            phi /= np.linalg.norm(phi)
            kchange = abs(keff-k_old)
            change = np.linalg.norm((phi-phi_old)/phi/(self.I*(self.L+1)))
            if LOUD:
                print('Change is',change,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS) or (kchange < 1e-10)
            count += 1
            phi_old = phi.copy()
            k_old = keff
            sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old[:,0,:]) 
            # sources = phi_old[:,0,:] * self.chiNuFission # np.sum(self.chiNuFission,axis=0)
        if self.track == 'both':
            return phi,keff,allmat_fis[1:],allmat_sca[1:]
        if self.track == 'scatter':
            return phi,keff,allmat_sca[1:]
        if self.track == 'fission':
            return phi,keff,allmat_fis[1:]
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

    def multi_group(G,N,mu,w,total,scatter,L,nuChiFission,I,delta,model,prediction,zones,tol=1e-08,MAX_ITS=100,LOUD=False):
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
            model: DJINN model
            prediction: I x G x G array of DJINN predicted sigma_s*phi 
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
            if count == 1:
                dj_pred = prediction.copy()
            else:
                enriched_width = zones[2] - zones[1] 
                # Separate out part going into DJINN
                enriched_ = phi_old[zones[1]:zones[2],0,:].copy()
                # Normalize by energy group (save norm to multiply back in)
                normed_ = np.linalg.norm(enriched_,axis=0)
                enriched_ /= normed_
                # Predict enriched zone y with normalized phi
                dj_pred = model.predict(enriched_).reshape(enriched_width,G,G)*normed_
            phi = np.zeros(phi_old.shape)
            # Iterate over all the energy groups
            for g in range(G):
                if (LOUD):
                    print("Inner Transport Iterations\n===================================")
                # Calculate the source term (takes downscattering matrix with update_q_djinn function)
                if g == 0:
                    q_tilde = nuChiFission[:,g].reshape(I,1) + func.update_q_djinn(N,mu,scatter,phi_old,g+1,G,g,dj_pred,zones)
                # Calculate the source term (downscattering and upscattering are separated)
                else:
                    q_tilde = nuChiFission[:,g].reshape(I,1) + func.update_q_djinn(N,mu,scatter,phi_old,g+1,G,g,dj_pred,zones) + func.update_q_djinn(N,mu,scatter,phi,0,g,g,dj_pred,zones)
                # Assigns new value to phi I x L+1
                phi[:,:,g] = eigen_djinn.one_group(N,mu,w,total[:,g],scatter[:,:,g,g],L,q_tilde,I,delta,phi_old[:,:,g],dj_pred[:,g,g],zones,tol=tol,MAX_ITS=MAX_ITS,LOUD=LOUD)
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
        from discrete1.util import sn
        from djinn import djinn
        #Initialize and normalize phi
        phi_old = np.random.rand(self.I,self.L+1,self.G)
        phi_old /= np.linalg.norm(phi_old)
        # Load DJINN Model
        model = djinn.load(model_name=model_name)
        # Set width of enriched zone 
        enriched_width = zones[2] - zones[1] 
        # Separate out part going into DJINN
        enriched_ = phi_old[zones[1]:zones[2],0,:].copy()
        # Normalize by energy group (save norm to multiply back in)
        normed_ = np.linalg.norm(enriched_,axis=0)
        enriched_ /= normed_
        # Predict enriched zone y with normalized phi
        djinn_y = model.predict(enriched_).reshape(enriched_width,self.G,self.G)*normed_
        #djinn_y = model.predict(phi_old[zones[1]:zones[2],0,:]).reshape(enriched_width,self.G,self.G)
        # Dimensionality Purposes
        if dimen:
            scatterSn = sn.propagate(xs=self.scatter,I=self.I,L=self.L,G=self.G,dtype='scatter')
            totalSn = sn.propagate(xs=self.total,I=self.I,G=self.G,dtype='total')
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
            phi = eigen_djinn.multi_group(self.G,self.N,self.mu,self.w,totalSn,scatterSn,self.L,sources,self.I,self.delta,model,djinn_y,zones,tol=1e-08,MAX_ITS=MAX_ITS,LOUD=False)
            # Calculate k-effective
            keff = np.linalg.norm(phi)
            # Normalize to 1
            phi /= np.linalg.norm(phi)
            # Check for convergence
            #k_change = np.fabs(keff - k_old)
            change = np.linalg.norm((phi-phi_old)/phi/(self.I*(self.L+1)))
            if LOUD:
                print('Change is',change,count,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            #Update phi
            phi_old = phi.copy()
            # Update y - Separate zone going into DJINN
            enriched_ = phi_old[zones[1]:zones[2],0,:].copy()
            # Normalize by energy group (save norm to multiply back in)
            normed_ = np.linalg.norm(enriched_,axis=0)
            enriched_ /= normed_
            # Predict enriched zone y with normalized phi
            djinn_y = model.predict(enriched_).reshape(enriched_width,self.G,self.G)*normed_
            #djinn_y = model.predict(phi_old[zones[1]:zones[2],0,:]).reshape(enriched_width,self.G,self.G)
            # Update sources
            sources = phi_old[:,0,:] * self.chiNuFission 
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
        self.delta = float(R)/I
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
            # djinn_scatter = np.zeros((self.I,self.L+1,self.G))
            # return np.zeros((self.I,self.G))
            lens = sn.length(self.splits['djinn'])
            return np.zeros((lens,self.G))
        elif self.dtype == 'scatter':# or self.dtype == 'both':
            return np.einsum('ijk,ik->ij',self.chiNuFission,phi_old[:,0,:]) 
            # scale = np.sum(normed*np.sum(sn.cat(self.scatter[:,0],self.splits['djinn']),axis=1),axis=1)/np.sum(djinn_scatter_ns,axis=1)
            # non_scale = sn.cat(np.ones((self.I,self.G)),self.splits['keep'])
            # djinn_scatter = (djinn_scatter_ns*scale).reshape(self.I,self.L+1,self.G)
            # djinn_scatter = np.concatenate((non_scale,np.expand_dims(scale,axis=1)*djinn_scatter_ns)).reshape(self.I,self.L+1,self.G)
            # Set sources for power iteration
            # if self.dtype == 'scatter':
                # sources = self.chiNuFission @ phi_old[0]
                # sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old[:,0,:]) 
        elif self.dtype == 'fission' or self.dtype == 'both':
            scale = np.sum(normed*np.sum(sn.cat(self.chiNuFission,self.splits['djinn']),axis=1),axis=1)/np.sum(djinn_fission_ns,axis=1)
            non_scale = sn.cat(np.ones((self.I,self.G)),self.splits['keep'])
            # print(scale)
            sources = np.concatenate((non_scale,np.expand_dims(scale,axis=1)*djinn_fission_ns)).reshape(self.I,self.G)
            # sources = np.concatenate((non_scale,djinn_fission_ns)).reshape(self.I,self.G)
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
        
    def one_group(self,N,mu,w,total,scatter,djinn_1g,L,external,I,delta,guess,tol=1e-08,MAX_ITS=100,LOUD=False):
        """ Arguments:
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
            phi: a I x L+1 array   """
        import numpy as np        
        from discrete1.util import sn
        converged = 0
        count = 1
        phi = np.zeros((I,L+1))
        phi_old = guess.copy()
        while not(converged):
            phi *= 0
            for n in range(N):
                non_weight_scatter = func.scattering_approx(L,mu[n])
                weight_scatter = w[n]*non_weight_scatter
                if self.dtype == 'scatter' or self.dtype == 'both':
                    # djinn_1g should be I'*L+1
                    multiplier = np.concatenate((sn.cat((scatter*phi_old),self.splits['keep']),np.expand_dims(djinn_1g,axis=1)))
                else:
                    multiplier = scatter * phi_old
                # temp_scat should be of length I
                temp_scat = np.sum(non_weight_scatter*multiplier,axis=1)
                psi_bottom = 0 # vacuum on LHS
                # Left to right
                for ii in range(I):
                    psi_top = (temp_scat[ii] + external[ii,n] + psi_bottom * func.total_add(total[ii], mu[n], delta,'right'))/(func.total_add(total[ii], mu[n], delta,'left'))
                    phi[ii,:] = phi[ii,:] + (weight_scatter * func.diamond_diff(psi_top,psi_bottom))
                    psi_bottom = psi_top
                # Reflective right to left
                for ii in range(I-1,-1,-1):
                    psi_top = psi_bottom
                    psi_bottom = (temp_scat[ii] + external[ii,n] + psi_top * func.total_add(total[ii], -mu[n], delta,'right'))/(func.total_add(total[ii], -mu[n], delta,'left'))
                    phi[ii,:] = phi[ii,:] +  (weight_scatter * func.diamond_diff(psi_top,psi_bottom)).flatten()
            change = np.linalg.norm((phi - phi_old)/phi/(I*(L+1)))
            converged = (change < tol) or (count >= MAX_ITS) 
            if LOUD:
                print('Iteration:',count,' Change:',change)
            count += 1
            phi_old = phi.copy()
        return phi
    
    def multi_group(self,G,N,mu,w,total,scatter,L,chiNuFission,model,I,delta,tol=1e-08,MAX_ITS=100,LOUD=False):
        # self,self.G,self.N,self.mu,self.w,self.total,self.scatter,self.L,sources,model_scatter,djinn_scatter,self.I,self.delta,
        """ Arguments:
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
            phi: a I x L+1 x G array  """
        import numpy as np
        # from discrete1.util import sn
        phi_old = np.zeros((I,L+1,G))
        converged = 0
        count = 1
        while not (converged):
            if LOUD:
                print('New Inner Loop Iteration\n======================================')
            phi = np.zeros(phi_old.shape)
            if self.dtype == 'scatter' or self.dtype == 'both':
                # Not going to be of size I'
                dj_pred = eigen_djinn_symm.multi_scale(self,phi_old,model,scatter,G,I,L)
                for g in range(G):
                    if (LOUD):
                        print("Inner Transport Iterations\n===================================")
                    # print(np.tile(np.expand_dims(nuChiFission[:,g],axis=1),(1,N)).shape)
                    # print(total.shape,scatter.shape,dj_pred.shape,chiNuFission.shape)
                    phi[:,:,g] = eigen_djinn_symm.one_group(self,N,mu,w,total[:,g],scatter[:,:,g,g],dj_pred[:,g],L,np.tile(np.expand_dims(chiNuFission[:,g],axis=1),(1,N)),I,delta,phi_old[:,:,g],tol=tol,MAX_ITS=MAX_ITS,LOUD=LOUD)
            elif self.dtype == 'fission':
                for g in range(G):
                    if (LOUD):
                        print("Inner Transport Iterations\n===================================")
                    if g == 0:
                        q_tilde = chiNuFission[:,g].reshape(I,1) + func.update_q(N,L,mu,scatter,phi_old,g+1,G,g)
                    else:
                        q_tilde = chiNuFission[:,g].reshape(I,1) + func.update_q(N,L,mu,scatter,phi_old,g+1,G,g) + func.update_q(N,L,mu,scatter,phi,0,g,g)
                    phi[:,:,g] = eigen_djinn_symm.one_group(self,N,mu,w,total[:,g],scatter[:,:,g,g],None,L,q_tilde,I,delta,tol=tol,MAX_ITS=MAX_ITS,LOUD=LOUD,guess=phi_old[:,:,g])
            change = np.linalg.norm((phi - phi_old)/phi/(I*(L+1)))
            if LOUD:
                print('Change is',change,count)
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
    
class inf_eigen:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L,enrich=False,matmul=False,track=False):
        self.G = G
        self.N = N
        self.mu = mu
        self.w = w
        self.total = total
        self.scatter = scatter
        self.chiNuFission = chiNuFission
        self.L = L
        self.enrich = enrich
        self.matmul = matmul
        self.track = track
   
    def one_group(self,N,mu,w,total,scatter,L,external,guess,pred_1g=None,tol=1e-08,MAX_ITS=100,LOUD=False):   
        ''' Infinite Eigenvalue Problem - One Group
        Arguments:
            N: Number of angles
            mu: vector of angles between -1 and 1
            w: vector of weights for the angles
            total: scalar of the total cross section
            scatter: L+1 array for the scattering moments
            L: Number of moments
            external: N x 2 array for the external sources
            guess: Initial guess of the scalar flux for a specific energy group (L+1)
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iterations and change between iterations, default is False
        Returns:
            phi: a L+1 array     '''
        import numpy as np
        
        converged = 0
        count = 1
        phi = np.zeros((L+1))
        phi_old = guess.copy()
        
        while not(converged):
            phi *= 0
            for n in range(N):
                non_weight_scatter = func.scattering_approx(L,mu[n])
                #weight_scatter = w[n]*non_weight_scatter
                if self.matmul:
                    temp_scat = np.sum(non_weight_scatter*pred_1g,axis=0)
                else:
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


    def multi_group(self,G,N,mu,w,total,scatter,L,nuChiFission,prediction=None,tol=1e-08,MAX_ITS=100,LOUD=False):
        ''' Infinite Eigenvalue Problem - Multigroup
        Arguments:
            G: number of energy groups
            N: number of angles
            mu: vector of angles between -1 and 1
            w: vector of weights for the angles
            total: G x 1 vector of the total cross section for each spatial cell and energy level
            scatter: L+1 x G array for the scattering of the spatial cell by moment and energy
            L: Number of moments 
            nuChiFission: Sources for each energy group
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a L+1 x G array     '''
        import numpy as np
        phi_old = np.zeros((L+1,G))
        converged = 0
        count = 1
        if self.track:
            if self.enrich:
                mymat = np.zeros((1,3,G))
            else:
                mymat = np.zeros((1,2,G))
        while not (converged):
            if LOUD:
                print('New Inner Loop Iteration\n======================================')
            phi = np.zeros(phi_old.shape)
            if self.matmul:
                if count == 1:
                    dj_pred = prediction.copy()
                else:
                    dj_pred = (scatter[0] @ phi_old[0]).reshape(L+1,G)
                    
            if self.track == 'scatter':
                if self.enrich:
                    track_temp = np.vstack((np.repeat(self.enrich,G),phi_old[0].flatten(),(scatter[0] @ phi_old[0]).flatten()))
                    mymat = np.vstack((mymat,track_temp.reshape(1,3,G)))
                else:
                    track_temp = np.vstack((phi_old[0].flatten(),(scatter[0] @ phi_old[0]).flatten()))
                    mymat = np.vstack((mymat,track_temp.reshape(1,2,G)))
                    
            for g in range(G):
                if (LOUD):
                    print("Inner Transport Iterations\n===================================")
                if self.matmul:
                    phi[:,g] = inf_eigen.one_group(self,N,mu,w,total[g],scatter[:,g,g],L,np.tile(nuChiFission[g],(N,1)),phi_old[:,g],dj_pred[:,g],tol=tol,MAX_ITS=MAX_ITS,LOUD=LOUD)
                # elif self.track == 'fission':
                #     if g == 0:
                #         q_tilde = multiplier[g] + func.update_q_inf(N,L,mu,scatter,phi_old,g+1,G,g)
                #     else:
                #         q_tilde = multiplier[g] + func.update_q_inf(N,L,mu,scatter,phi_old,g+1,G,g) + func.update_q_inf(N,L,mu,scatter,phi,0,g,g)
                #     phi[:,g] = inf_eigen.one_group(self,N,mu,w,total[g],scatter[:,g,g],L,q_tilde,phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS,LOUD=LOUD)
                else:
                    if g == 0:
                        q_tilde = nuChiFission[g] + func.update_q_inf(N,L,mu,scatter,phi_old,g+1,G,g)
                    else:
                        q_tilde = nuChiFission[g] + func.update_q_inf(N,L,mu,scatter,phi_old,g+1,G,g) + func.update_q_inf(N,L,mu,scatter,phi,0,g,g)
                    phi[:,g] = inf_eigen.one_group(self,N,mu,w,total[g],scatter[:,g,g],L,q_tilde,phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS,LOUD=LOUD)
            change = np.linalg.norm((phi - phi_old)/phi/((L+1)))
            if LOUD:
                print('Change is',change,count)
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        if self.track == 'scatter':
            return phi,mymat[1:]
        return phi

    def transport(self,predy=None,tol=1e-12,MAX_ITS=100,LOUD=True):
        ''' Infinite Eigenvalue Problem
        Arguments:
            self:
            tol: tolerance of convergence, default is 1e-12
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is True
        Returns:
            phi: a L+1 x G array     '''
        import numpy as np
        # Initialize and normalize phi
        phi_old = np.random.rand(self.L+1,self.G)
        # k_old = np.linalg.norm(phi_old)
        phi_old /= np.linalg.norm(phi_old)
        # Calculate original source terms
        sources = self.chiNuFission @ phi_old[0] 
        if self.matmul:
            predy = (self.scatter @ phi_old[0]).reshape(self.L+1,self.G)
        if self.track:
            if self.enrich:
                allmat = np.zeros((1,3,self.G))
            else:
                allmat = np.zeros((1,2,self.G))
        if self.track == 'fission':
            if self.enrich:
                track_temp = np.vstack((np.repeat(self.enrich,self.G),phi_old[0].flatten(),(self.chiNuFission @ phi_old[0]).flatten()))
                allmat = np.vstack((allmat,track_temp.reshape(1,3,self.G)))
            else:
                track_temp = np.vstack((phi_old[0].flatten(),(self.chiNuFission @ phi_old[0]).flatten()))
                allmat = np.vstack((allmat,track_temp.reshape(1,2,self.G)))
        # Parameters for convergence/count
        converged = 0
        count = 1
        while not (converged):
            if LOUD:
                print('Outer Transport Iteration\n===================================')
            # Calculate phi
            if self.track == 'scatter':
                phi,tempmat = inf_eigen.multi_group(self,self.G,self.N,self.mu,self.w,self.total,self.scatter.reshape(self.L+1,self.G,self.G),self.L,sources,predy,tol=1e-08,MAX_ITS=MAX_ITS,LOUD=False)
                allmat = np.vstack((allmat,tempmat))
            else:
                phi = inf_eigen.multi_group(self,self.G,self.N,self.mu,self.w,self.total,self.scatter.reshape(self.L+1,self.G,self.G),self.L,sources,predy,tol=1e-08,MAX_ITS=MAX_ITS,LOUD=False)
            # Normalize phi and k-effective
            keff = np.linalg.norm(phi)
            phi /= np.linalg.norm(phi)
            # Check for convergence
            change = np.linalg.norm((phi-phi_old)/phi/((self.L+1)))
            if LOUD:
                print('Change is',change,count,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS) #or (abs(k_old - keff) < 1e-10)
            count += 1
            # Update phi and source terms
            phi_old = phi.copy()
            # k_old = keff
            if self.matmul:
                predy = (self.scatter @ phi_old[0]).reshape(self.L+1,self.G)
            # sources = phi_old[0,:] * self.chiNuFission 
            sources = self.chiNuFission @ phi_old[0]
            if self.track == 'fission':
                if self.enrich:
                    track_temp = np.vstack((np.repeat(self.enrich,self.G),phi_old[0].flatten(),(self.chiNuFission @ phi_old[0]).flatten()))
                    allmat = np.vstack((allmat,track_temp.reshape(1,3,self.G)))
                else:
                    track_temp = np.vstack((phi_old[0].flatten(),(self.chiNuFission @ phi_old[0]).flatten()))
                    allmat = np.vstack((allmat,track_temp.reshape(1,2,self.G)))
        if self.track:
            return phi,keff,allmat[1:]
        return phi,keff    

class inf_eigen_djinn:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L,dtype='scatter',enrich=False):
        self.G = G
        self.N = N
        self.mu = mu
        self.w = w
        self.total = total
        self.scatter = scatter
        self.chiNuFission = chiNuFission
        self.L = L
        self.dtype = dtype
        self.enrich = enrich
        
   
    def one_group_scatter(N,mu,w,total,djinn_1g,L,external,guess,tol=1e-08,MAX_ITS=100,LOUD=False):   
        ''' Infinite DJINN One Group Calculation - y scatter
        Arguments:
            N: Number of angles
            mu: vector of angles between -1 and 1
            w: vector of weights for the angles
            total: value of the total cross section for each spatial cell
            djinn_1g: L+1 list of DJINN predicted sigma_s*phi
            L: Number of moments
            external: 2 x N array for the external sources
            guess: Initial guess of the scalar flux for a specific energy group (L+1)
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

    def one_group_fission(self,N,mu,w,total,scatter,L,external,guess,tol=1e-08,MAX_ITS=100,LOUD=False):   
        """ Infinite Eigenvalue Problem - One Group
        Arguments:
            N: Number of angles
            mu: vector of angles between -1 and 1
            w: vector of weights for the angles
            total: scalar of the total cross section
            scatter: L+1 array for the scattering moments
            L: Number of moments
            external: N x 2 array for the external sources
            guess: Initial guess of the scalar flux for a specific energy group (L+1)
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iterations and change between iterations, default is False
        Returns:
            phi: a L+1 array   """
        import numpy as np
        
        converged = 0
        count = 1
        phi = np.zeros((L+1))
        phi_old = guess.copy()
        
        while not(converged):
            phi *= 0
            for n in range(N):
                non_weight_scatter = func.scattering_approx(L,mu[n])
                #weight_scatter = w[n]*non_weight_scatter
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

    def multi_group(self,G,N,mu,w,total,scatter,L,nuChiFission,model,djinn_prediction,tol=1e-08,MAX_ITS=100,LOUD=False):
        """ Infinite DJINN Multigroup Calculation (Source Iteration) - y
        Arguments:
            G: number of energy groups
            N: number of angles
            mu: vector of angles between -1 and 1
            w: vector of weights for the angles
            total: G x 1 vector of the total cross section for each spatial cell and energy level
            L: Number of moments
            nuChiFission: G x 1 external source
            model_name: DJINN model
            djinn_prediction: Outer Iteration Predicted sigma_s*phi
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a L+1 x G array
        """
        import numpy as np        
        # Initialize phi
        phi_old = np.zeros((L+1,G))
        # Parameters for convergence/count
        converged = 0
        count = 1

        while not (converged):
            if LOUD:
                print('New Inner Loop Iteration\n======================================')
            # zero out new phi
            phi = np.zeros(phi_old.shape)
            # Update the DJINN predictions with new phi
            if self.dtype == 'scatter' or self.dtype == 'both':
                if(np.sum(phi_old) == 0):
                    dj_pred = np.zeros((L+1,G))
                else:
                    normed = phi_old.copy()
                    if self.enrich:
                        dj_pred_ns = model.predict(np.array([self.enrich]+list(normed.flatten()))).flatten()
                    else:
                        dj_pred_ns = model.predict(normed)
                    scale = np.sum(normed[0]*np.sum(scatter,axis=0))/np.sum(dj_pred_ns)
                    dj_pred = (dj_pred_ns*scale).reshape(L+1,G)
                # if self.track:
                #     # dj_pred, phi_old (normalized), uh3_scatter @ phi_old
                #     track_temp = np.vstack((dj_pred.flatten(),normed.flatten(),scatter @ phi_old[0]))
                #     mymat = np.vstack((mymat,track_temp.reshape(1,3,87)))
                # Iterate over all the energy groups
                for g in range(G):
                    if (LOUD):
                        print("Inner Transport Iterations\n===================================")
                    # Assigns new value to phi L+1 
                    phi[:,g] = inf_eigen_djinn.one_group_scatter(N,mu,w,total[g],dj_pred[:,g],L,np.tile(nuChiFission[g],(N,1)),phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS,LOUD=LOUD)
            elif self.dtype == 'fission':
                for g in range(G):
                    if g == 0:
                        q_tilde = nuChiFission[g] + func.update_q_inf(N,L,mu,scatter,phi_old,g+1,G,g)
                    else:
                        q_tilde = nuChiFission[g] + func.update_q_inf(N,L,mu,scatter,phi_old,g+1,G,g) + func.update_q_inf(N,L,mu,scatter,phi,0,g,g)
                    phi[:,g] = inf_eigen_djinn.one_group_fission(self,N,mu,w,total[g],scatter[:,g,g],L,q_tilde,phi_old[:,g],tol=tol,MAX_ITS=MAX_ITS,LOUD=LOUD)
            # Check for convergence
            change = np.linalg.norm((phi - phi_old)/phi/((L+1)))
            if LOUD:
                print('Change is',change,count)
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            # Update phi and repeat
            phi_old = phi.copy()
        # if self.track:
        #     return phi,mymat[1:]
        return phi

    def transport(self,model_name,tol=1e-12,MAX_ITS=100,LOUD=True):
        """ Infinite DJINN Multigroup Calculation (Power Iteration) - y
        Arguments:
            model_name: file location of DJINN model
                if both: list of file locations (scatter,fission)
            tol: tolerance of convergence, default is 1e-12
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a L+1 x G array  """
        import numpy as np
        from djinn import djinn
        # Initialize and normalize phi
        phi_old = np.random.rand(self.L+1,self.G)
        k_old = np.linalg.norm(phi_old)
        phi_old /= np.linalg.norm(phi_old)
        # Load DJINN model
        if self.dtype == 'both':    
            model_scatter = djinn.load(model_name=model_name[0])
            model_fission = djinn.load(model_name=model_name[1])
        elif self.dtype == 'scatter':
            model_scatter = djinn.load(model_name=model_name)
        elif self.dtype == 'fission':
            model_fission = djinn.load(model_name=model_name)
        # Set lengths and predict model
        if self.enrich:
            if self.dtype == 'scatter' or self.dtype == 'both':
                djinn_scatter_ns = model_scatter.predict(np.array([self.enrich]+list(phi_old.flatten()))).flatten()
            if self.dtype == 'fission' or self.dtype == 'both':
                djinn_fission_ns = model_fission.predict(np.array([self.enrich]+list(phi_old.flatten()))).flatten()
        else:
            if self.dtype == 'scatter' or self.dtype =='both':
                djinn_scatter_ns = model_scatter.predict(phi_old)
            if self.dtype == 'fission' or self.dtype == 'both':
                djinn_fission_ns = model_fission.predict(phi_old)
                
        # Predict sigma_s*phi from initialized phi
        # phi_old = np.load('mydata/djinn_test/true_phi.npy') # Hot start
        # Setting length of vector to include label or not
        # if self.enrich:
        #     djinn_y_ns = model.predict(np.array([self.enrich]+list(phi_old.flatten()))).flatten()
        # else:
        #     djinn_y_ns = model.predict(phi_old)
            
        if np.sum(phi_old) == 0 or self.dtype == 'fission':
            djinn_scatter = np.zeros((self.L+1,self.G))
        elif self.dtype == 'scatter' or self.dtype == 'both':
            scale = np.sum(phi_old[0]*np.sum(self.scatter,axis=0))/np.sum(djinn_scatter_ns)
            djinn_scatter = (djinn_scatter_ns*scale).reshape(self.L+1,self.G)
            # Set sources for power iteration
            if self.dtype == 'scatter':
                sources = self.chiNuFission @ phi_old[0]
        if self.dtype == 'fission' or self.dtype == 'both':
            scale = np.sum(phi_old[0]*np.sum(self.chiNuFission,axis=0))/np.sum(djinn_fission_ns)
            sources = (djinn_fission_ns*scale).flatten()
            self.scatter = self.scatter.reshape(self.L+1,self.G,self.G)
    
        # if self.track:
        #     allmat = np.zeros((1,3,87))
        # Parameters for convergence/count
        converged = 0
        count = 1
        while not (converged):
            if LOUD:
                print('Outer Transport Iteration\n===================================')
            # Calculate phi with original sources
            # if self.track:
            #     phi,tempmat = inf_eigen_djinn.multi_group(self,self.G,self.N,self.mu,self.w,self.total,self.scatter,self.L,sources,model,djinn_y,tol=1e-08,MAX_ITS=MAX_ITS,LOUD=False)
            #     allmat = np.vstack((allmat,tempmat))
            # else:
            phi = inf_eigen_djinn.multi_group(self,self.G,self.N,self.mu,self.w,self.total,self.scatter,self.L,sources,model_scatter,djinn_scatter,tol=1e-08,MAX_ITS=MAX_ITS,LOUD=False)
            # Calculate k-effective
            keff = np.linalg.norm(phi)
            # Normalize to 1
            phi /= np.linalg.norm(phi)
            # Check for convergence
            change = np.linalg.norm((phi-phi_old)/phi/((self.L+1)))
            if LOUD:
                print('Change is',change,count,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS) or (abs(k_old - keff) < 1e-10)
            count += 1
            # Update phi
            phi_old = phi.copy()
            k_old = keff
            
            # Assign lengths and predict new matrix multiplication
            if self.enrich:
                if self.dtype == 'scatter' or self.dtype == 'both':
                    djinn_scatter_ns = model_scatter.predict(np.array([self.enrich]+list(phi_old.flatten()))).flatten()
                if self.dtype == 'fission' or self.dtype == 'both':
                    djinn_fission_ns = model_fission.predict(np.array([self.enrich]+list(phi_old.flatten()))).flatten()
            else:
                if self.dtype == 'scatter' or self.dtype == 'both':
                    djinn_scatter_ns = model_scatter.predict(phi_old)
                if self.dtype == 'fission' or self.dtype == 'both':
                    djinn_fission_ns = model_fission.predict(phi_old)                    
            # If there is all zeros, return zeros
            if np.sum(phi_old) == 0 or self.dtype == 'fission':
                djinn_scatter = np.zeros((self.L+1,self.G))
                if self.dtype != 'fission':
                    sources = np.zeros((self.G))
            # Scaling after DJINN
            elif self.dtype == 'scatter' or self.dtype == 'both':
                scale = np.sum(phi_old[0]*np.sum(self.scatter,axis=0))/np.sum(djinn_scatter_ns)
                djinn_scatter = (djinn_scatter_ns*scale).reshape(self.L+1,self.G)
                # Set sources for power iteration
                if self.dtype == 'scatter':
                    sources = self.chiNuFission @ phi_old[0]
            if self.dtype == 'fission' or self.dtype == 'both':
                scale = np.sum(phi_old[0]*np.sum(self.chiNuFission,axis=0))/np.sum(djinn_fission_ns)
                sources = (djinn_fission_ns*scale).flatten()
                self.scatter = self.scatter.reshape(self.L+1,self.G,self.G)
                
            # if self.enrich:
            #     djinn_y_ns = model.predict(np.array([self.enrich]+list(phi_old.flatten()))).flatten()
            # else:
            #     djinn_y_ns = model.predict(phi_old)
            # if np.sum(phi_old) == 0 or self.dtype == 'fission':
            #     djinn_y = np.zeros((self.L+1,self.G))
            # elif self.dtype == 'scatter':
            #     scale = np.sum(phi_old[0]*np.sum(self.scatter,axis=0))/np.sum(djinn_y_ns)
            #     djinn_y = (djinn_y_ns*scale).reshape(self.L+1,self.G)
            #     # Update sources
            #     sources = self.chiNuFission @ phi_old[0]
            # if self.dtype == 'fission':
            #     scale = np.sum(phi_old[0]*np.sum(self.chiNuFission,axis=0))/np.sum(djinn_y_ns)
            #     sources = (djinn_y_ns*scale).flatten()
                
            # print(np.sum(phi_old[0]*np.sum(self.scatter,axis=0)),np.sum(self.scatter @ phi_old[0]))
        # if self.track:
        #     return phi,keff,allmat[1:]   
        return phi,keff
    
