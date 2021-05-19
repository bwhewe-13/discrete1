""" Generating Cross Sections, etc. for Sn code and display """

from .chemistry import NumberDensity

import numpy as np
import pkg_resources

# Used for determining path to cross sections
DATA_PATH = pkg_resources.resource_filename('discrete1','xs/')


class XSGenerate087:
    """ Generating the total, fission, scatter cross sections """
    __allowed = ("enrich") # Currently only keyword
    __compounds = ("UH3","HDPE","SS440","U","C") # Materials that we can use
    __fissioning = ("UH3","U") # Materials that fission
    # Using the first temperature
    __temp = '00'

    def __init__(self,compound,**kwargs):
        assert (compound in self.__class__.__compounds), "Compound not allowed, available: UH3, HDPE, SS440, U"
        self.compound = compound
        # Kwargs
        self.enrich = 0.0; 
        # Make sure user inputs right kwargs
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, available: enrich, energy" 
            setattr(self, key, value)

    def cross_section(self):
        # Dictionary of Number Densities
        nd_dict = NumberDensity(self.compound,enrich=self.enrich).run()
        # Total Cross Section List
        total = [np.load(DATA_PATH + '{}/vecTotal.npy'.format(ii))[eval(self.__class__.__temp)] for ii in nd_dict.keys()]
        total_xs = sum([total[count]*nd for count,nd in enumerate(nd_dict.values())])
        # Scatter Cross Section List
        scatter = [np.load(DATA_PATH + '{}/scatter_0{}.npy'.format(ii,self.__class__.__temp))[0] for ii in nd_dict.keys()]
        scatter_xs = sum([scatter[count]*nd for count,nd in enumerate(nd_dict.values())])
        # Check if it is a fissioning material
        fission_xs = np.zeros((scatter_xs.shape))
        if self.compound in self.__class__.__fissioning:
            fission = [np.load(DATA_PATH + '{}/nufission_0{}.npy'.format(ii,self.__class__.__temp))[0] for ii in nd_dict.keys()]
            fission_xs = sum([fission[count]*nd for count,nd in enumerate(nd_dict.values())])

        return total_xs, scatter_xs.T, fission_xs.T


class XSGenerate618:
    """ Generating the total, fission, scatter cross sections 
    The number densities do not have to be calculated  
    """

    def cross_section(enrich):
        # Scattering
        pu239_scatter = np.load(DATA_PATH + 'pu239/scatter_000.npy')
        pu240_scatter = np.load(DATA_PATH + 'pu240/scatter_000.npy')
        hdpe_scatter = np.load(DATA_PATH + 'hdpe/scatter_000.npy')
        enrich_scatter = pu239_scatter * (1 - enrich) + pu240_scatter * enrich
        scatter_xs = [hdpe_scatter,enrich_scatter,pu240_scatter]

        del pu239_scatter, pu240_scatter, hdpe_scatter, enrich_scatter

        # Fission
        pu239_fission = np.load(DATA_PATH + 'pu239/nufission_000.npy')
        pu240_fission = np.load(DATA_PATH + 'pu240/nufission_000.npy')
        hdpe_fission = np.zeros(pu239_fission.shape)
        enrich_fission = pu239_fission * (1 - enrich) + pu240_fission * enrich
        fission_xs = [hdpe_fission,enrich_fission,pu240_fission]

        del pu239_fission, pu240_fission, hdpe_fission, enrich_fission

        # Total
        pu239_total = np.load(DATA_PATH + 'pu239/vecTotal.npy')
        pu240_total = np.load(DATA_PATH + 'pu240/vecTotal.npy')
        hdpe_total = np.load(DATA_PATH + 'hdpe/vecTotal.npy')
        enrich_total = pu239_total * (1 - enrich) + pu240_total * enrich
        total_xs = [hdpe_total,enrich_total,pu240_total]
        
        del pu239_total, pu240_total, hdpe_total, enrich_total        
        
        return total_xs, scatter_xs, fission_xs