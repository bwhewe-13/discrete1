# Stopped at Page 27 out of 58 on benchmarks pdf
def key(print_all=True,one_group=False):
    ''' key for calling different functions '''
    print('============================================')
    print('|| This deals with benchmark problems,')
    print('|| with class calls listed below.')
    print('|| In each class there are the explain,')
    print('|| variables, and run functions')
    print('============================================')
    if(print_all or one_group):
        print('|| ONE-ENERGY GROUP BENCHMARKS')
        print('============================================================')
        print('|| g1_pu239_ex1: One-Energy Group Isotropic Cross Section\n||')
        print('|| g1_pu239_ex2: One-Energy Group Isotropic Cross Section\n||')
        print('|| g1_u235_ex1: One-Energy Group Isotropic Cross Section\n||')
        print('|| g1_u235_ex2: One-Energy Group Isotropic Cross Section\n||')
        print('|| g1_u235_ex3: One-Energy Group Isotropic Cross Section\n||')
        print('|| g1_u235_ex4: One-Energy Group Isotropic Cross Section\n||')

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
