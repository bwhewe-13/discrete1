""" 
Generating Cross Sections, etc. for Sn code and display
"""

from .chemistry import NumberDensity

import numpy as np
import pkg_resources

# Used for determining path to cross sections
DATA_PATH = pkg_resources.resource_filename('discrete1','xs/')


class XSGenerate:
    """ Generating the total, fission, scatter cross sections """
    __allowed = ("enrich") # Currently only keyword
    __compounds = ("UH3","HDPE","SS440") # Materials that we can use
    __fissioning = ("UH3") # Materials that fission
    # Using the first temperature
    __temp = '00'

    def __init__(self,compound,**kwargs):
        assert (compound in self.__class__.__compounds), "Compound not allowed, available: UH3, HDPE, SS440, Pu"
        self.compound = compound
        # Kwargs
        self.enrich = 0.0; 
        # Make sure user inputs right kwargs
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, available: enrich, energy" 
            setattr(self, key, value)

    def cross_section(self):
        # Dictionary of Number Densities
        nd_dict = NumberDensity.run(self.compound,self.enrich)
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

        return total_xs, scatter_xs, fission_xs



