class eigen:
    def __init__(self,G,N,mu,w,total,scatter,chiNuFission,L,R,I):
        self.G = G
        self.N = N
        self.mu = mu
        self.w = w
        self.total = total
        self.full_scatter = scatter
        self.full_chiNuFission = chiNuFission
        self.I = I
        self.inv_delta = float(I)/R
        # self.enrich = enrich
        
        
    def one_group(self,total,scatter,smult,external,guess,tol=1e-08,MAX_ITS=100):
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
        clibrary = ctypes.cdll.LoadLibrary('./discrete1/data/cfunctions.so')
        sweep = clibrary.sweep
        converged = 0
        count = 1        
        phi_old = guess.copy()
        half_total = 0.5*total.copy()
        external = external.astype('float64')
        while not(converged):
            phi = np.zeros((self.I),dtype='float64')
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
        from discrete1.setup import func

        phi_old = guess.copy()

        converged = 0; count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)
            smult = np.einsum('ijk,ik->ij',scatter,phi_old)
            for g in range(self.G):
                if g == 0:
                    q_tilde = nuChiFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g)
                else:
                    q_tilde = nuChiFission[:,g] + func.update_q(scatter,phi_old,g+1,self.G,g) + func.update_q(scatter,phi,0,g,g)
                phi[:,g] = eigen.one_group(self,total[:,g],scatter[:,g,g],smult[:,g],q_tilde,tol=tol,MAX_ITS=MAX_ITS,guess=phi_old[:,g])

            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()

        return phi
            
    def transport(self,problem,rank,solution=None,distance=[45,35,20],group=70,tol=1e-12,MAX_ITS=100):
        """ Arguments:
            tol: tolerance of convergence, default is 1e-08
            MAX_ITS: maximum iterations allowed, default is 100
            LOUD: prints the iteration number and the change between iterations, default is False
        Returns:
            phi: a I x G array    """        
        import numpy as np
        from discrete1.setup import func,problem2

        print(distance)
        
        phi_old = func.initial_flux(problem,group)
        if solution is None:
            solution = phi_old.copy()
        self.scatter,self.chiNuFission = func.low_rank_svd(solution,self.full_scatter,self.full_chiNuFission,problem,rank,distance)
        # self.scatter,self.chiNuFission = func.low_rank_svd_squeeze(618,70,70,distance=distance)

        print('scatter',self.scatter.shape)
        
        sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old) 
        converged = 0; count = 1
        while not (converged):
            print('Outer Transport Iteration {}\n==================================='.format(count))
            phi = eigen.multi_group(self,self.total,self.scatter,sources,phi_old,tol=1e-08,MAX_ITS=MAX_ITS)
            keff = np.linalg.norm(phi)
            phi /= keff
                        
            change = np.linalg.norm((phi-phi_old)/phi/(self.I))
            print('Change is',change,'Keff is',keff)
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1

            phi_old = phi.copy()
            sources = np.einsum('ijk,ik->ij',self.chiNuFission,phi_old) 
        return phi,keff


