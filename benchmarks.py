def key(print_all=False,one_group=False,two_group=False,three_group=False,six_group=False):
    if sum([one_group,two_group,three_group,six_group]) == 0:
        print_all = True
    ''' key for calling different functions '''
    print('============================================')
    print('|| This deals with benchmark problems,')
    print('|| with class calls listed below.')
    print('|| In each class there are the explain,')
    print('|| variables, and run functions')
    print('============================================')
    if(print_all or one_group):
        print('|| ONE-ENERGY GROUP BENCHMARKS')
        print('=================================================================')
        print('|| g1_pu239_ex1: One-Energy Group Isotropic Cross Section\n||')
        print('|| g1_pu239_ex2: One-Energy Group Isotropic Cross Section\n||')
        print('|| g1_u235_ex1: One-Energy Group Isotropic Cross Section\n||')
        print('|| g1_u235_ex2: One-Energy Group Isotropic Cross Section\n||')
        print('|| g1_u235_ex3: One-Energy Group Isotropic Cross Section\n||')
        print('|| g1_u235_ex4: One-Energy Group Isotropic Cross Section\n||')
    if(print_all or two_group):
        print('|| TWO-ENERGY GROUP BENCHMARKS')
        print('=================================================================')
        print('|| g2_pu239_ex1: Two-Energy Group Isotropic Cross Section\n||')
        print('|| g2_urr_ex1: Two-Energy Group Isotropic Cross Section\n||')
        print('|| g2_ual_ex1: Two-Energy Group Isotropic Cross Section\n||')
        print('|| g2_urr2m_ex1: Two-Medium, Two-Energy Group Isotropic\n||')
    if(print_all or three_group):
        print('|| THREE-ENERGY GROUP BENCHMARKS')
        print('=================================================================')
        print('|| g3_rr_ex1: Three-Energy Group Isotropic Cross Section\n||')
    if(print_all or six_group):
        print('|| SIX-ENERGY GROUP BENCHMARKS')
        print('=================================================================')
        print('|| g6_rr-ex1: Six-Energy Group Isotropic Cross Section\n||')
        
class g1_pu239_ex1:
    def __init__(self,N=None,R=None,I=None):
        self.N = N
        self.R = R
        self.I = I
    
    def explain(self=None):
        print('===============================================')
        print('|| One-Energy Group Isotropic Cross Section')
        print('|| Slab: Critical Radius = 1.853722')
        print('|| Pu-239, nu = 3.24')
        print('|| k_inf = 2.612903')
        print('===============================================')
        
    def variables(self,dim=False):
        # dim will put cross sections into appropriate dimensions
        import numpy as np
        G = 1; L = 0
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w)
        chiNuFission = np.array([3.24*0.0816])
        total = np.array([0.32640])
        scatter = np.array([0.225216])
        if dim:
            fission_ = np.tile(chiNuFission,(self.I,G))
            scatter_ = np.tile(scatter,(self.I,L+1,G,G))
            total_ = np.tile(total,(self.I,G))
            return G,self.N,mu,w,total_,scatter_,fission_,L
        return G,self.N,mu,w,total,scatter,chiNuFission,L
    
    def run(self,crit=False,inf=False):
        import discrete1.slab as s
        if inf:
            problem = s.inf_eigen(*g1_pu239_ex1.variables(self))
            return problem.transport()        
        if crit:
            rad = 1.853722*2
        else:
            rad = self.R
        # get first 8 variables 
        first_args = list(g1_pu239_ex1.variables(self,dim=True))
        # Add in R and I
        full_args = tuple(first_args + [rad,self.I])
        # Initialize problem and run
        problem = s.eigen(*full_args)
        return problem.transport()

class g1_pu239_ex2:
    def __init__(self,N=None,R=None,I=None):
        self.N = N
        self.R = R
        self.I = I
    
    def explain(self=None):
        print('===============================================')
        print('|| One-Energy Group Isotropic Cross Section')
        print('|| Slab: Critical Radius = 2.256751')
        print('|| Pu-239, nu = 2.84')
        print('|| k_inf = 2.290323')
        print('===============================================')
        
    def variables(self,dim=False):
        # dim will put cross sections into appropriate dimensions
        import numpy as np
        G = 1; L = 0
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w)
        chiNuFission = np.array([2.84*0.0816])
        total = np.array([0.32640])
        scatter = np.array([0.225216])
        if dim:
            fission_ = np.tile(chiNuFission,(self.I,G))
            scatter_ = np.tile(scatter,(self.I,L+1,G,G))
            total_ = np.tile(total,(self.I,G))
            return G,self.N,mu,w,total_,scatter_,fission_,L
        return G,self.N,mu,w,total,scatter,chiNuFission,L
    
    def run(self,crit=False,inf=False):
        import discrete1.slab as s
        if inf:
            problem = s.inf_eigen(*g1_pu239_ex2.variables(self))
            return problem.transport()        
        if crit:
            rad = 2.256751*2
        else:
            rad = self.R
        # get first 8 variables 
        first_args = list(g1_pu239_ex2.variables(self,dim=True))
        # Add in R and I
        full_args = tuple(first_args + [rad,self.I])
        # Initialize problem and run
        problem = s.eigen(*full_args)
        return problem.transport()

class g1_u235_ex1:
    def __init__(self,N=None,R=None,I=None):
        self.N = N
        self.R = R
        self.I = I
    
    def explain(self=None):
        print('===============================================')
        print('|| One-Energy Group Isotropic Cross Section')
        print('|| Slab: Critical Radius = 2.872934')
        print('|| U-235, nu = 2.70')
        print('|| k_inf = 2.25')
        print('===============================================')
        
    def variables(self,dim=False):
        # dim will put cross sections into appropriate dimensions
        import numpy as np
        G = 1; L = 0
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w)
        chiNuFission = np.array([2.70*0.065280])
        total = np.array([0.32640])
        scatter = np.array([0.248064])
        if dim:
            fission_ = np.tile(chiNuFission,(self.I,G))
            scatter_ = np.tile(scatter,(self.I,L+1,G,G))
            total_ = np.tile(total,(self.I,G))
            return G,self.N,mu,w,total_,scatter_,fission_,L
        return G,self.N,mu,w,total,scatter,chiNuFission,L
    
    def run(self,crit=False,inf=False):
        import discrete1.slab as s
        if inf:
            problem = s.inf_eigen(*g1_u235_ex1.variables(self))
            return problem.transport()        
        if crit:
            rad = 2.872934*2
        else:
            rad = self.R
        # get first 8 variables 
        first_args = list(g1_u235_ex1.variables(self,dim=True))
        # Add in R and I
        full_args = tuple(first_args + [rad,self.I])
        # Initialize problem and run
        problem = s.eigen(*full_args)
        return problem.transport()

class g1_u235_ex2:
    def __init__(self,N=None,R=None,I=None):
        self.N = N
        self.R = R
        self.I = I
    
    def explain(self=None):
        print('===============================================')
        print('|| One-Energy Group Isotropic Cross Section')
        print('|| Slab: Critical Radius = N/A')
        print('|| U-235, nu = 2.797101')
        print('|| k_inf = 2.330917 (Default)')
        print('===============================================')
        
    def variables(self,dim=False):
        # dim will put cross sections into appropriate dimensions
        import numpy as np
        G = 1; L = 0
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w)
        chiNuFission = np.array([2.797101*0.065280])
        total = np.array([0.32640])
        scatter = np.array([0.248064])
        if dim:
            fission_ = np.tile(chiNuFission,(self.I,G))
            scatter_ = np.tile(scatter,(self.I,L+1,G,G))
            total_ = np.tile(total,(self.I,G))
            return G,self.N,mu,w,total_,scatter_,fission_,L
        return G,self.N,mu,w,total,scatter,chiNuFission,L
    
    def run(self,crit=False,inf=True):
        import discrete1.slab as s
        if inf:
            problem = s.inf_eigen(*g1_u235_ex2.variables(self))
            return problem.transport()        
        if crit:
            print('slab critical unknown')
            rad = 2.872934*2
        else:
            rad = self.R
        # get first 8 variables 
        first_args = list(g1_u235_ex2.variables(self,dim=True))
        # Add in R and I
        full_args = tuple(first_args + [rad,self.I])
        # Initialize problem and run
        problem = s.eigen(*full_args)
        return problem.transport()

class g1_u235_ex3:
    def __init__(self,N=None,R=None,I=None):
        self.N = N
        self.R = R
        self.I = I
    
    def explain(self=None):
        print('===============================================')
        print('|| One-Energy Group Isotropic Cross Section')
        print('|| Slab: Critical Radius = N/A')
        print('|| U-235, nu = 2.707308')
        print('|| k_inf = 2.256083 (Default)')
        print('===============================================')
        
    def variables(self,dim=False):
        # dim will put cross sections into appropriate dimensions
        import numpy as np
        G = 1; L = 0
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w)
        chiNuFission = np.array([2.707308*0.065280])
        total = np.array([0.32640])
        scatter = np.array([0.248064])
        if dim:
            fission_ = np.tile(chiNuFission,(self.I,G))
            scatter_ = np.tile(scatter,(self.I,L+1,G,G))
            total_ = np.tile(total,(self.I,G))
            return G,self.N,mu,w,total_,scatter_,fission_,L
        return G,self.N,mu,w,total,scatter,chiNuFission,L
    
    def run(self,crit=False,inf=True):
        import discrete1.slab as s
        if inf:
            problem = s.inf_eigen(*g1_u235_ex3.variables(self))
            return problem.transport()        
        if crit:
            print('slab critical unknown')
            rad = 2.872934*2
        else:
            rad = self.R
        # get first 8 variables 
        first_args = list(g1_u235_ex3.variables(self,dim=True))
        # Add in R and I
        full_args = tuple(first_args + [rad,self.I])
        # Initialize problem and run
        problem = s.eigen(*full_args)
        return problem.transport()

class g1_u235_ex4:
    def __init__(self,N=None,R=None,I=None):
        self.N = N
        self.R = R
        self.I = I
    
    def explain(self=None):
        print('===============================================')
        print('|| One-Energy Group Isotropic Cross Section')
        print('|| Slab: Critical Radius = N/A')
        print('|| U-235, nu = 2.679198')
        print('|| k_inf = 2.232667 (Default)')
        print('===============================================')
        
    def variables(self,dim=False):
        # dim will put cross sections into appropriate dimensions
        import numpy as np
        G = 1; L = 0
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w)
        chiNuFission = np.array([2.679198*0.065280])
        total = np.array([0.32640])
        scatter = np.array([0.248064])
        if dim:
            fission_ = np.tile(chiNuFission,(self.I,G))
            scatter_ = np.tile(scatter,(self.I,L+1,G,G))
            total_ = np.tile(total,(self.I,G))
            return G,self.N,mu,w,total_,scatter_,fission_,L
        return G,self.N,mu,w,total,scatter,chiNuFission,L
    
    def run(self,crit=False,inf=True):
        import discrete1.slab as s
        if inf:
            problem = s.inf_eigen(*g1_u235_ex4.variables(self))
            return problem.transport()        
        if crit:
            print('slab critical unknown')
            rad = 2.872934*2
        else:
            rad = self.R
        # get first 8 variables 
        first_args = list(g1_u235_ex4.variables(self,dim=True))
        # Add in R and I
        full_args = tuple(first_args + [rad,self.I])
        # Initialize problem and run
        problem = s.eigen(*full_args)
        return problem.transport()

class g2_pu239_ex1:
    def __init__(self,N=None,R=None,I=None):
        self.N = N
        self.R = R
        self.I = I
    
    def explain(self=None):
        print('===============================================')
        print('|| Two-Energy Group Isotropic Cross Section')
        print('|| Slab: Critical Radius = 1.795602')
        print('|| Pu-239, nu1 = 2.93, nu2 = 3.10')
        print('|| k_inf = 2.683767 (Default)')
        print('===============================================')
        
    def variables(self,dim=False):
        # dim will put cross sections into appropriate dimensions
        import numpy as np
        G = 2; L = 0
        if self.N is None:
            self.N = 8
        mu,w = np.polynomial.legendre.leggauss(self.N)        
        w /= np.sum(w)
        # chi = np.array([0.425,0.575])
        chi = np.array([[0.425],[0.575]])
        nu = np.array([[2.93,3.1]])
        fission = np.array([[0.08544,0.0936]])
        chiNuFission = chi @ (nu * fission)
        total = np.array([0.3360,0.2208])
        scatter = np.array([[0.23616,0],
                            [0.0432,0.0792]])
        if dim:
            fission_ = np.tile(chiNuFission,tuple([self.I]+2*[1]))
            scatter_ = np.tile(scatter.T,tuple([self.I,L+1]+2*[1]))
            total_ = np.tile(total,(self.I,1))
            return G,self.N,mu,w,total_,scatter_,fission_,L
        return G,self.N,mu,w,total,scatter.T,chiNuFission,L
    
    def run(self,crit=False,inf=True):
        import discrete1.slab as s
        if inf:
            problem = s.inf_eigen(*g2_pu239_ex1.variables(self))
            return problem.transport()        
        if crit:
            rad = 1.795602*2
        else:
            rad = self.R
        # get first 8 variables 
        first_args = list(g2_pu239_ex1.variables(self,dim=True))
        # Add in R and I
        full_args = tuple(first_args + [rad,self.I])
        # Initialize problem and run
        problem = s.eigen(*full_args)
        return problem.transport()

class quickieTT:
    """ Testing rouge discrete1 script """
    def __init__(self,N=None,R=None,I=None):
        self.N = N
        self.R = R
        self.I = I
        
    def variables(self,dim=False):
        # dim will put cross sections into appropriate dimensions
        import numpy as np
        G = 2; L = 0
        if self.N is None:
            self.N = 8
        mu,w = np.polynomial.legendre.leggauss(self.N)        
        w /= np.sum(w)
        
        mu = mu[int(self.N*0.5):]
        w = w[int(self.N*0.5):]
        self.N = int(self.N*0.5)
        
        # chi = np.array([0.425,0.575])
        chi = np.array([[0.425],[0.575]])
        nu = np.array([[2.93,3.1]])
        fission = np.array([[0.08544,0.0936]])
        chiNuFission = chi @ (nu * fission)
        total = np.array([0.3360,0.2208])
        scatter = np.array([[0.23616,0],
                            [0.0432,0.0792]])
        if dim:
            fission_ = np.tile(chiNuFission,tuple([self.I]+2*[1]))
            scatter_ = np.tile(scatter.T,tuple([self.I,L+1]+2*[1]))
            total_ = np.tile(total,(self.I,1))
            return G,self.N,mu,w,total_,scatter_[:,0],fission_,L
        return G,self.N,mu,w,total,scatter.T,chiNuFission,L
    
    def run(self,crit=False,inf=True):
        import discrete1.rouge as s
        if inf:
            problem = s.inf_eigen(*quickieTT.variables(self))
            return problem.transport()        
        if crit:
            rad = 1.795602
        else:
            rad = self.R
        # get first 8 variables 
        first_args = list(quickieTT.variables(self,dim=True))
        # Add in R and I
        full_args = tuple(first_args + [rad,self.I])
        # Initialize problem and run
        problem = s.eigen_symm(*full_args)
        return problem.transport()

class g2_urr_ex1:
    def __init__(self,N=None,R=None,I=None):
        self.N = N
        self.R = R
        self.I = I
        
    def explain(self=None):
        print('===============================================')
        print('|| Two-Energy Group Isotropic Cross Section')
        print('|| Slab: Critical Radius = Unknown')
        print('|| Uranium Research Reactor (B), nu = 2.5')
        print('|| k_inf = 1.365821 (Default)')
        print('===============================================')
        
    def variables(self,dim=False):
        # dim will put cross sections into appropriate dimensions
        import numpy as np
        G = 2; L = 0
        if self.N is None:
            self.N = 8
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w)
        chi = np.array([[0],[1]])
        nu = np.array([[2.5,2.5]])
        fission = np.array([[0.029564,0.000836]])
        chiNuFission = chi @ (nu * fission)
        total = np.array([2.9727,0.88721])
        scatter = np.array([[2.9183,0.000767],
                            [0.04635,0.83892]])
        if dim:
            fission_ = np.tile(chiNuFission,tuple([self.I]+2*[1]))
            scatter_ = np.tile(scatter.T,tuple([self.I,L+1]+2*[1]))
            total_ = np.tile(total,(self.I,1))
            return G,self.N,mu,w,total_,scatter_,fission_,L
        return G,self.N,mu,w,total,scatter.T,chiNuFission,L
    
    def run(self,crit=False,inf=True):
        import discrete1.slab as s
        if inf:
            problem = s.inf_eigen(*g2_urr_ex1.variables(self))
            return problem.transport()        
        if crit:
            print('slab critical unknown')
            rad = 2.5*2
        else:
            rad = self.R
        # get first 8 variables 
        first_args = list(g2_urr_ex1.variables(self,dim=True))
        # Add in R and I
        full_args = tuple(first_args + [rad,self.I])
        # Initialize problem and run
        problem = s.eigen(*full_args)
        return problem.transport()

class g2_ual_ex1:
    def __init__(self,N=None,R=None,I=None):
        self.N = N
        self.R = R
        self.I = I
    
    def explain(self=None):
        print('===============================================')
        print('|| Two-Energy Group Isotropic Cross Section')
        print('|| Slab: Critical Radius = 7.830630')
        print('|| U-Al, nu1 = 2.83, nu2 = 0.0')
        print('|| k_inf = 2.661745 (Default)')
        print('===============================================')
        
    def variables(self,dim=False):
        # dim will put cross sections into appropriate dimensions
        import numpy as np
        G = 2; L = 0
        if self.N is None:
            self.N = 8
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w)
        chi = np.array([[0],[1]])
        nu = np.array([[2.83,0]])
        fission = np.array([[0.06070636042,0]])
        chiNuFission = chi @ (nu * fission)
        total = np.array([1.27698,0.26817])
        scatter = np.array([[1.21313,0],
                            [0.020432,0.247516]])
        if dim:
            fission_ = np.tile(chiNuFission,tuple([self.I]+2*[1]))
            scatter_ = np.tile(scatter.T,tuple([self.I,L+1]+2*[1]))
            total_ = np.tile(total,(self.I,1))
            return G,self.N,mu,w,total_,scatter_,fission_,L
        return G,self.N,mu,w,total,scatter.T,chiNuFission,L
    
    def run(self,crit=False,inf=True):
        import discrete1.slab as s
        if inf:
            problem = s.inf_eigen(*g2_ual_ex1.variables(self))
            return problem.transport()        
        if crit:
            rad = 7.830630*2
        else:
            rad = self.R
        # get first 8 variables 
        first_args = list(g2_ual_ex1.variables(self,dim=True))
        # Add in R and I
        full_args = tuple(first_args + [rad,self.I])
        # Initialize problem and run
        problem = s.eigen(*full_args)
        return problem.transport()

class g2_urr2m_ex1:
    def __init__(self,N=None,R=None,delta=None):
        self.N = N
        self.R = R
        self.delta = delta
        
    def explain(self=None):
        print('===================================================')
        print('|| Two-Medium, Two-Energy Group Isotropic ')
        print('|| Slab: Critical Radius = Unknown (No Infinite)')
        print('|| Uranium Research Reactor (B), nu = 2.5')
        print('|| H20 Reflector (A)')
        print('|| Distance (R) is a list not a float')
        print('|| Cell width is used instead of number of cells')
        print('===================================================')
        
    def variables(self):
        import numpy as np
        from discrete1.util import sn_tools
        G = 2; L = 0
        if self.N is None:
            self.N = 8
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w)
        # Research Reactor (B) Data
        chi_rr = np.array([[0],[1]])
        nu_rr = np.array([[2.5,2.5]])
        fission_rr = np.array([[0.029564,0.000836]])
        chiNuFission_rr = chi_rr @ (nu_rr * fission_rr)
        total_rr = np.array([2.9727,0.88721])
        scatter_rr = np.array([[2.9183,0.000767],
                            [0.04635,0.83892]])
        # H20 Reflector (A) Data
        chiNuFission_h20 = np.array([[0,0],[0,0]])
        total_h20 = np.array([2.9865,0.88798])
        scatter_h20 = np.array([[2.9676,0.000336],
                                [0.04749,0.83975]])
        
        xs_total = [total_h20,total_rr,total_h20]
        xs_fission = [chiNuFission_h20,chiNuFission_rr,chiNuFission_h20]
        xs_scatter = [scatter_h20.T,scatter_rr.T,scatter_h20.T]
        layers = [int(ii/self.delta) for ii in self.R]
        self.I = int(sum(layers))
    
        scatter_ = sn_tools.mixed_propagate(xs_scatter,layers,G=G,L=L,dtype='scatter')
        fission_ = sn_tools.mixed_propagate(xs_fission,layers,G=G,dtype='fission2')
        total_ = sn_tools.mixed_propagate(xs_total,layers,G=G)
        
        return G,self.N,mu,w,total_,scatter_,fission_,L
        # return G,self.N,mu,w,total,scatter.T,chiNuFission,L
    
    def run(self,crit=False):
        import discrete1.slab as s
        if crit:
            self.R = [1.126152,6.696802*2,1.126152]
        rad = sum(self.R)
        # get first 8 variables 
        first_args = list(g2_urr2m_ex1.variables(self))
        # Add in R and I
        full_args = tuple(first_args + [rad,self.I])
        # Initialize problem and run
        problem = s.eigen(*full_args)
        return problem.transport()
    
class g3_rr_ex1:
    def __init__(self,N=None,R=None,I=None):
        self.N = N
        self.R = R
        self.I = I
    
    def explain(self=None):
        print('======================================================')
        print('|| Three-Energy Group Isotropic Cross Section')
        print('|| Slab: Critical Radius = N/A')
        print('|| Research Reactor, nu1 = 2.0, nu2 = 2.5. nu3 = 3.0')
        print('|| k_inf = 1.60 (Default)')
        print('======================================================')
        
    def variables(self,dim=False):
        # dim will put cross sections into appropriate dimensions
        import numpy as np
        G = 3; L = 0
        if self.N is None:
            self.N = 8
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w)
        chi = np.array([[0],[0.04],[0.96]])
        nu = np.array([[2.0,2.5,3.0]])
        fission = np.array([[0.90,0.060,0.006]])
        chiNuFission = chi @ (nu * fission)
        total = np.array([3.10,0.975,0.240])
        scatter = np.array([[2.0,0.0,0.0],
                            [0.275,0.6,0.0],
                            [0.033,0.171,0.024]])
        if dim:
            fission_ = np.tile(chiNuFission,tuple([self.I,1,1]))
            scatter_ = np.tile(scatter.T,tuple([self.I,L+1,1,1]))
            total_ = np.tile(total,(self.I,1))
            return G,self.N,mu,w,total_,scatter_,fission_,L
        return G,self.N,mu,w,total,scatter.T,chiNuFission.transpose(),L
    
    def run(self,crit=False,inf=True):
        import discrete1.slab as s
        if inf:
            problem = s.inf_eigen(*g3_rr_ex1.variables(self))
            return problem.transport()        
        if crit:
            print('slab critical unknown')
            rad = 1.5*2
        else:
            rad = self.R
        # get first 8 variables 
        first_args = list(g3_rr_ex1.variables(self,dim=True))
        # Add in R and I
        full_args = tuple(first_args + [rad,self.I])
        # Initialize problem and run
        problem = s.eigen(*full_args)
        return problem.transport()    

class g6_rr_ex1:
    def __init__(self,N=None,R=None,I=None):
        self.N = N
        self.R = R
        self.I = I
    
    def explain(self=None):
        print('======================================================')
        print('|| Six-Energy Group Isotropic Cross Section')
        print('|| Slab: Critical Radius = N/A')
        print('|| Research Reactor, nu = [3,2.5,2,2,2.5,3]')
        print('|| k_inf = 1.60 (Default)')
        print('======================================================')
        
    def variables(self,dim=False):
        # dim will put cross sections into appropriate dimensions
        import numpy as np
        G = 6; L = 0
        if self.N is None:
            self.N = 8
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w)
        chi = np.array([[0.48],[0.02],[0.0],[0.0],[0.02],[0.48]])
        nu = np.array([[3.0,2.5,2.0,2.0,2.5,3.0]])
        fission = np.array([[0.006,0.060,0.90,0.90,0.060,0.006]])
        chiNuFission = chi @ (nu * fission)
        total = np.array([0.240,0.975,3.10,3.10,0.975,0.240])
        scatter = np.array([[0.024,0.171,0.033,0,0,0],
                            [0,0.60,0.275,0,0,0],
                            [0,0,2.0,0,0,0],
                            [0,0,0,2.0,0,0],
                            [0,0,0,0.275,0.60,0],
                            [0,0,0,0.033,0.171,0.024]])
        if dim:
            fission_ = np.tile(chiNuFission,tuple([self.I]+2*[1]))
            scatter_ = np.tile(scatter.T,tuple([self.I,L+1]+2*[1]))
            total_ = np.tile(total,(self.I,1))
            return G,self.N,mu,w,total_,scatter_,fission_,L
        return G,self.N,mu,w,total,scatter.T,chiNuFission,L
    
    def run(self,crit=False,inf=True):
        import discrete1.slab as s
        if inf:
            problem = s.inf_eigen(*g6_rr_ex1.variables(self))
            return problem.transport()        
        if crit:
            print('slab critical unknown')
            rad = 1.5*2
        else:
            rad = self.R
        # get first 8 variables 
        first_args = list(g6_rr_ex1.variables(self,dim=True))
        # Add in R and I
        full_args = tuple(first_args + [rad,self.I])
        # Initialize problem and run
        problem = s.eigen(*full_args)
        return problem.transport()    