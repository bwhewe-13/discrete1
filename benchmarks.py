def key():
    ''' key for calling different functions '''
    print('============================================')
    print('|| This deals with benchmark problems,')
    print('|| with class calls listed below.')
    print('|| In each class there are the explain,')
    print('|| variables, and run functions')
    print('============================================')
    print('|| g1_pu239_ex1: One-Energy Group Isotropic Cross Section\n||')

class g1_pu239_ex1:
    def __init__(self,N,R,I):
        self.N = N
        self.R = R
        self.I = I
    
    def explain():
        print('===============================================')
        print('|| One-Energy Group Isotropic Cross Section')
        print('|| Slab: Critical Radius = 1.853722')
        print('|| Nu = 3.24')
        print('|| k_inf = 2.612903')
        print('===============================================')
        
    def variables():
        import numpy as np
        G = 1; L = 0
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w)
        chiNuFission = np.array([3.24*0.0816])
        total = np.array([0.32640])
        scatter = np.array([0.225216])
        return G,self.N,mu,w,total,scatter,chiNuFission,L
    
    def run(crit=False,inf=False):
        import discrete1.slab as s        
        if inf:
            problem = s.inf_eigen(*g1_pu239_ex1.variables())
            return problem.transport()
        
        if crit:
            rad = 1.853722*2
        else:
            rad = self.R
        delta = float(rad)/self.I
        
        first_args = list(*g1_pu239_ex1.variables())
        full_args = tuple(first_args + [self.I,delta])
        problem = s.eigen(*full_args)
        return problem.transport()
        
        
        
        