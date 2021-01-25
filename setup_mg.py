
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
        down_scat = np.array([(1/(3*D_g[ii]) - sigma_a[ii]) - sigma_ds[ii] for ii in range(self.G-1)])

        scatter_vals = np.diag(down_scat,-1)
        np.fill_diagonal(scatter_vals,1/(3*D_g) - sigma_a)
        scatter_ = np.tile(scatter_vals,(I,1,1))

        source_vals = [1e12,0,0,0]
        source_ = np.tile(source_vals,(I,1))

        fission_ = np.zeros((scatter_.shape))

        return self.G,self.N,mu,w,total_,scatter_,fission_,source_,I,1/delta



class UraniumHDPE:
    def __init__(self,G,N):
        self.G = G
        self.N = N

    def variables(self,enrich,reduced=False):
        import numpy as np
        from .util import chem,sn

        if self.G != 87:
            reduced = True

        distance = [45,35,40,35,45]
        # distance = [ii*0.5 for ii in distance]
        density_uh3 = 10.95; density_ch3 = 0.97
        
        uh3_density = chem.density_list('UH3',density_uh3,enrich)
        hdpe_density = chem.density_list('CH3',density_ch3)
        uh3_238_density = chem.density_list('U^238H3',density_uh3)
    
        # Loading Cross Section Data
        spec_temp = '00'
        # Scattering Cross Section
        u235scatter = np.load('mydata/u235/scatter_0{}.npy'.format(spec_temp))[0]
        u238scatter = np.load('mydata/u238/scatter_0{}.npy'.format(spec_temp))[0]
        h1scatter = np.load('mydata/h1/scatter_0{}.npy'.format(spec_temp))[0]
        c12scatter = np.load('mydata/cnat/scatter_0{}.npy'.format(spec_temp))[0]
    
        uh3_scatter = uh3_density[0]*u235scatter + uh3_density[1]*u238scatter + uh3_density[2]*h1scatter
        hdpe_scatter = hdpe_density[0]*c12scatter + hdpe_density[1]*h1scatter
        uh3_238_scatter = uh3_238_density[0]*u238scatter + uh3_238_density[1]*h1scatter
    
        # Total Cross Section
        u235total = np.load('mydata/u235/vecTotal.npy')[eval(spec_temp)]
        u238total = np.load('mydata/u238/vecTotal.npy')[eval(spec_temp)]
        h1total = np.load('mydata/h1/vecTotal.npy')[eval(spec_temp)]
        c12total = np.load('mydata/cnat/vecTotal.npy')[eval(spec_temp)]
    
        uh3_total = uh3_density[0]*u235total + uh3_density[1]*u238total + uh3_density[2]*h1total
        hdpe_total = hdpe_density[0]*c12total + hdpe_density[1]*h1total
        uh3_238_total = uh3_238_density[0]*u238total + uh3_238_density[1]*h1total
    
        # Fission Cross Section
        u235fission = np.load('mydata/u235/nufission_0{}.npy'.format(spec_temp))[0]
        u238fission = np.load('mydata/u238/nufission_0{}.npy'.format(spec_temp))[0]
    
        uh3_fission = uh3_density[0]*u235fission + uh3_density[1]*u238fission

        uh3_238_fission = uh3_238_density[0]*u238fission
        hdpe_fission = np.zeros((self.G,self.G))

        # Cross section layers

        xs_scatter = [hdpe_scatter.T,uh3_scatter.T,uh3_238_scatter.T,uh3_scatter.T,hdpe_scatter.T]
        # xs_scatter = [hdpe_scatter,uh3_scatter,uh3_238_scatter,uh3_scatter,hdpe_scatter]
        xs_total = [hdpe_total,uh3_total,uh3_238_total,uh3_total,hdpe_total]
        xs_fission = [hdpe_fission.T,uh3_fission.T,uh3_238_fission.T,uh3_fission.T,hdpe_fission.T]
        # xs_fission = [hdpe_fission,uh3_fission,uh3_238_fission,uh3_fission,hdpe_fission]

        energy_grid = np.load('discrete1/data/energyGrid.npy')
        if reduced:
            print("Reduced")
            hdpe_total,hdpe_scatter,hdpe_fission = sn.group_reduction(self.G,energy_grid,
                total=hdpe_total,scatter=hdpe_scatter.T,fission=hdpe_fission.T)
            uh3_total,uh3_scatter,uh3_fission = sn.group_reduction(self.G,energy_grid,
                total=uh3_total,scatter=uh3_scatter.T,fission=uh3_fission.T)
            uh3_238_total,uh3_238_scatter,uh3_238_fission = sn.group_reduction(self.G,energy_grid,
                total=uh3_238_total,scatter=uh3_238_scatter.T,fission=uh3_238_fission.T)
            # Don't need to be transposed
            xs_scatter = [hdpe_scatter,uh3_scatter,uh3_238_scatter]
            xs_total = [hdpe_total,uh3_total,uh3_238_total]
            xs_fission = [hdpe_fission,uh3_fission,uh3_238_fission]

        # Setting up eigenvalue equation
        N = 8; L = 0; R = sum(distance); #G = self.G
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w); I = 1000; delta = float(R)/I

        layers = [int(ii/delta) for ii in distance]
        # I = int(sum(layers)); delta = float(R)/I    

        # One Unit Source in 14.1 MeV group from left
        g = np.argmin(abs(energy_grid-14.1E6))
        source = np.zeros((I,self.G))
        source[0,g] = 1
        

        scatter_ = sn.mixed_propagate(xs_scatter,layers,G=self.G,L=L,dtype='scatter')
        fission_ = sn.mixed_propagate(xs_fission,layers,G=self.G,dtype='fission2')
        total_ = sn.mixed_propagate(xs_total,layers,G=self.G)
        
        return self.G,self.N,mu,w,total_,scatter_[:,0],fission_,source,I,delta

    def scatter_fission(self,enrich,reduced=None):
        _,_,_,_,_,scatter,fission,_,_,_ = UraniumHDPE.variables(self,enrich,reduced=reduced)
        return scatter,fission
