""" Classes used for Approximating phi * sigma

Available Methods:
    - DJINN
    - Autoencoder + DJINN
    - Autoencoder (squeeze)
    - SVD
"""

import numpy as np
from djinn import djinn


class DJ:
    __allowed = ("double","focus")

    def __init__(self,model_name,atype,**kwargs):
        """ For functions related to using DJINN and approximating phi * sigma
        Attributes:
            model_name: string or list of strings of DJINN models
            atype: string of approxmation type (can be 'fission','scatter',
                 or 'both')
        kwargs:
            double: if using multiple models for one operation, default False
            focus: if looking at the reflecting or fuel material
        """
        # Attributes
        self.model_name = model_name
        self.atype = atype 
        # kwargs
        self.double = False; self.focus = 'fuel'
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, available: double, focus" 
            setattr(self, key, value)

    def load_model(self):
        """ Load the DJINN models """
        if self.double:
            self.fuel_scatter,self.fuel_fission = Tools.djinn_load_driver(self.model_name[0],self.atype)
            self.refl_scatter,self.refl_fission = Tools.djinn_load_driver(self.model_name[1],self.atype)
        elif self.focus == 'reflector':
            self.refl_scatter,self.refl_fission = Tools.djinn_load_driver(self.model_name,self.atype)
            self.fuel_scatter = None; self.fuel_fission = None
        elif self.focus == 'fuel':
            self.refl_scatter = None; self.refl_fission = None
            self.fuel_scatter,self.fuel_fission = Tools.djinn_load_driver(self.model_name,self.atype)
        print('DJINN Models Loaded')

    # def

    

class Tools:

    def djinn_load_driver(model_,atype):
        model_fission = None; model_scatter = None
        if atype == 'both':
            model_scatter = djinn.load(model_name=model_[0])
            model_fission = djinn.load(model_name=model_[1])
        elif atype == 'scatter':
            model_scatter = djinn.load(model_name=model_)
        elif atype == 'fission':
            model_fission = djinn.load(model_name=model_)
        return model_scatter,model_fission




