
def selection(problem,G,N):
    if problem == 'reeds':
        select = Reeds(G,N)
    elif problem == 'four_group':
        select = FourGroup(G,N)
    return select.variables()

class Reeds:
    def __init__(self,G,N):
        self.G = 1
        self.N = N

    def variables(self):
        import numpy as np
        
        L = 0; R = 16.; I = 1000
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w); 

        delta = R/I

        boundaries = [slice(0,int(2/delta)),slice(int(2/delta),int(3/delta)),
            slice(int(3/delta),int(5/delta)),slice(int(5/delta),int(6/delta)),
            slice(int(6/delta),int(10/delta)),slice(int(10/delta),int(11/delta)),
            slice(int(11/delta),int(13/delta)),slice(int(13/delta),int(14/delta)),
            slice(int(14/delta),int(16/delta))]
        
        total_ = np.zeros((I)); total_vals = [10,10,0,5,50,5,0,10,10]
        scatter_ = np.zeros((I)); scatter_vals = [9.9,9.9,0,0,0,0,0,9.9,9.9]
        source_ = np.zeros((I)); source_vals = [0,1,0,0,50,0,0,1,0]

        for ii in range(len(boundaries)):
            total_[boundaries[ii]] = total_vals[ii]
            scatter_[boundaries[ii]] = scatter_vals[ii]
            source_[boundaries[ii]] = source_vals[ii]

        fission_ = np.zeros((scatter_.shape))

        return self.G,self.N,mu,w,total_[:,None],scatter_[:,None,None],fission_[:,None,None],source_[:,None],I,1/delta

class FourGroup:
    def __init__(self,G,N):
        self.G = 4
        self.N = N

    def variables(self):
        import numpy as np

        L = 0; R = 5.; I = 1000
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w); 

        delta = R/I
        
        sigma_a = np.array([0.00490, 0.00280, 0.03050, 0.12100])
        sigma_ds = np.array([0.08310,0.05850,0.06510])
        D_g = np.array([2.16200,1.08700,0.63200,0.35400])
            
        total_ = np.tile(1/(3*D_g),(I,1))
        down_scat = np.array([(1/(3*D_g[ii]) - sigma_a[ii]) - sigma_ds[ii] for ii in range(G-1)])

        scatter_vals = np.diag(down_scat,-1)
        np.fill_diagonal(scatter_vals,1/(3*D_g) - sigma_a)
        scatter_ = np.tile(scatter_vals,(I,1,1))

        source_vals = [1e12,0,0,0]
        source_ = np.tile(source_vals,(I,1))

        fission_ = np.zeros((scatter_.shape))

        return G,N,mu,w,total_,scatter_,fission_,source_,I,1/delta

