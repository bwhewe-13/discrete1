""" Classes used for Approximating phi * sigma

Available Methods:
    - DJINN
    - Autoencoder + DJINN
    - Autoencoder (squeeze)
    - SVD
"""

from .keigenvalue import Problem1,Problem2

import numpy as np
from djinn import djinn
from tensorflow import keras


class DJ:
    __allowed = ("double","focus","label")

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
        self.double = False; self.focus = 'fuel'; self.label = False
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, available: double, focus, label" 
            setattr(self, key, value)

    def load_model(self):
        """ Load the DJINN models """
        if self.double:
            self.fuel_scatter,self.fuel_fission = Tools.djinn_load_driver(self.model_name[0],self.atype)
            self.refl_scatter,self.refl_fission = Tools.djinn_load_driver(self.model_name[1],self.atype)
        elif self.focus == 'fuel':
            self.fuel_scatter,self.fuel_fission = Tools.djinn_load_driver(self.model_name,self.atype)
            self.refl_scatter = None; self.refl_fission = None
        elif self.focus == 'reflector':
            self.fuel_scatter = None; self.fuel_fission = None
            self.refl_scatter,self.refl_fission = Tools.djinn_load_driver(self.model_name,self.atype)
        print('DJINN Models Loaded')

    def load_problem(self,problem,enrich,orient='orig'):
        """ Loading all the extra Data to run DJINN """
        # DJ.load_model() # Initialize DJINN models
        if problem in ['hdpe','ss440']:
            self.labels,self.splits = Problem1.labeling(problem,enrich,orient)
            _,_,_,_,_,self.scatter_full,self.fission_full,_,_ = Problem1.steady(problem,enrich,orient)
        elif problem in ['pu']:
            self.labels,self.splits = Problem2.labeling('hdpe',enrich,orient)
            _,_,_,_,_,self.scatter_full,self.fission_full,_,_ = Problem2.steady('hdpe',enrich,orient)

        self.scatter_scale = np.sum(self.scatter_full,axis=1)
        self.fission_scale = np.sum(self.fission_full,axis=1)
        print('Problem Loaded')

    def predict_scatter(self,phi):
        """ predict phi * sigma_s """
        if np.sum(phi) == 0:
            return phi

        # Separate into refl and fuel
        phi_hot = Tools.concat(phi,self.splits['fuel'])
        phi_cold = Tools.concat(phi,self.splits['refl'])

        # Get scaling factors
        if self.double or self.focus == 'fuel':
            scale_hot = phi_hot * Tools.concat(self.scatter_scale,self.splits['fuel'])
        if self.double or self.focus == 'refl':
            scale_cold = phi_cold * Tools.concat(self.scatter_scale,self.splits['refl'])

        # Labeling the necessary models
        if self.label and self.focus == 'fuel':
            phi_hot = np.hstack((Tools.concat(self.labels,self.splits['fuel'])[:,None],phi_hot))
        if self.label and (self.double or self.focus == 'refl'):
            phi_cold = np.hstack((Tools.concat(self.labels,self.splits['refl'])[:,None],phi_cold))

        # Predict and scale
        if self.double or self.focus == 'fuel':
            phi_hot = self.fuel_scatter.predict(phi_hot)
            phi_hot = scale_hot/np.sum(phi_hot,axis=1)[:,None] * phi_hot
        if self.double or self.focus == 'refl':
            phi_cold = self.refl_scatter.predict(phi_cold)
            phi_cold = scale_cold/np.sum(phi_cold,axis=1)[:,None] * phi_cold

        # Repopulate matrix
        if np.array_equal(phi_hot,Tools.concat(phi,self.splits['fuel'])):
            phi_hot = np.einsum('ijk,ik->ij',Tools.concat(self.scatter_full,self.splits['fuel']),phi_hot)
        if np.array_equal(phi_cold,Tools.concat(phi,self.splits['refl'])):
            phi_cold = np.einsum('ijk,ik->ij',Tools.concat(self.scatter_full,self.splits['refl']),phi_cold)
        phi = Tools.repopulate(phi_hot,phi_cold,self.splits)

        return phi

    def predict_fission(self,phi):
        """ predict phi * sigma_f """
        if np.sum(phi) == 0 or self.focus == 'refl':
            return phi

        # Separate into refl and fuel
        phi_hot = Tools.concat(phi,self.splits['fuel'])
        phi_cold = Tools.concat(phi,self.splits['refl'])

        # Get scaling factors
        scale_hot = np.sum(phi_hot * Tools.concat(self.fission_scale,self.splits['fuel']),axis=1)

        # Labeling the necessary models
        if self.label:
            phi_hot = np.hstack((Tools.concat(self.labels,self.splits['fuel'])[:,None],phi_hot))

        # Predict and scale        
        phi_hot = self.fuel_fission.predict(phi_hot)
        phi_hot = (scale_hot/np.sum(phi_hot,axis=1))[:,None] * phi_hot

        # Repopulate matrix
        phi_cold = np.einsum('ijk,ik->ij',Tools.concat(self.fission_full,self.splits['refl']),phi_cold)
        phi = Tools.repopulate(phi_hot,phi_cold,self.splits)

        return phi


class AE:
    __allowed = ("double","focus")

    def __init__(self,model_name,atype,transform='cuberoot',**kwargs):
        """ For functions related to using full autoencoder (squeeze)
        Attributes:
            model_name: string or list of strings of DJINN models
            atype: string of approxmation type (can be 'fission','scatter',
                 or 'phi')
            transform: string of transformation, 'cuberoot','minmax'
        kwargs:
            double: if using multiple models for one operation, default False
            focus: if looking at the reflecting or fuel material
        """
        # Attributes
        self.model_name = model_name
        self.atype = atype 
        self.transform = transform
        # kwargs
        self.double = False; self.focus = 'fuel'; 
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, available: double, focus" 
            setattr(self, key, value)


    def load_model(self):
        """ Load the autoencoder model """
        if self.double:
            self.fuel_model = keras.models.load_model('{}_autoencoder.h5'.format(self.model_name[0]))
            self.refl_model = keras.models.load_model('{}_autoencoder.h5'.format(self.model_name[1]))
        elif self.focus == 'fuel':
            self.fuel_model = keras.models.load_model('{}_autoencoder.h5'.format(self.model_name))
            self.refl_model = None
        elif self.focus == 'reflector':
            self.fuel_model = None;
            self.refl_model = keras.models.load_model('{}_autoencoder.h5'.format(self.model_name))
        print('Autoencoder Loaded')
        

    def load_problem(self,problem,enrich,orient='orig'):
        """ Loading all the extra Data to run DJINN """
        if problem in ['hdpe','ss440']:
            _,self.splits = Problem1.labeling(problem,enrich,orient)
        elif problem in ['pu']:
            _,self.splits = Problem2.labeling('hdpe',enrich,orient)
        print('Problem Loaded')


    def squeeze(self,phi):
        """ predict phi * sigma_s """
        if np.sum(phi) == 0:
            return phi

        # Separate into refl and fuel
        phi_hot = Tools.concat(phi,self.splits['fuel'])
        phi_cold = Tools.concat(phi,self.splits['refl'])

        # Working with fuel
        if self.double or self.focus == 'fuel':
            phi_hot,maxi_hot,mini_hot = Tools.transformation(phi_hot,self.transform)     # Transform
            scale_hot = np.sum(phi_hot,axis=1)                                           # Scale
            phi_hot = self.fuel_model.predict(phi_hot)                                   # Predict
            phi_hot = scale_hot/np.sum(phi_hot,axis=1)[:,None] * phi_hot                 # Unscale
            phi_hot = Tools.detransformation(phi_hot,maxi_hot,mini_hot,self.transform)   # Untransform

        # Working with refl
        if self.double or self.focus == 'refl':
            phi_cold,maxi_cold,mini_cold = Tools.transformation(phi_cold,self.transform)   # Transform
            scale_cold = np.sum(phi_cold,axis=1)                                           # Scale
            phi_cold = self.refl_model.predict(phi_cold)                                   # Predict
            phi_cold = scale_cold/np.sum(phi_cold,axis=1)[:,None] * phi_cold               # Unscale
            phi_cold = Tools.detransformation(phi_cold,maxi_cold,mini_cold,self.transform) # Untransform

        # Repopulate matrix
        phi = Tools.repopulate(phi_hot,phi_cold,self.splits)

        return phi
    

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

    def concat(lst,splits):
        return np.concatenate(([lst[ii] for ii in splits]))   

    def repopulate(hot,cold,splits):
        """ Repopulating Phi * Sigma Matrix 
        hot: fuel material
        cold: reflector material
        splits: dictionary of splits    
        """
        phi = np.zeros((len(hot) + len(cold),hot.shape[1]))

        for mat,ele in zip(['refl','fuel'],[cold,hot]):
            start = 0
            for ind in splits[mat]:
                dist = ind.stop - ind.start + start
                phi[ind] = ele[start:dist]
                start = dist
        return phi

    def transformation(matrix,ttype='cuberoot'):
        reduced = matrix.copy()
        maxi = None; mini = None
        if ttype == 'minmax':
            maxi = np.max(reduced,axis=1)
            mini = np.max(reduced,axis=1)
            reduced = (reduced - mini[:,None]) / (maxi - mini)[:,None]
        elif ttype == 'cuberoot':
            reduced = reduced**(1/3)

        reduced[np.isnan(reduced)] = 0; reduced[np.isinf(reduced)] = 0
        return reduced,maxi,mini

    def detransformation(matrix,maxi,mini,ttype='cuberoot'):
        reduced = matrix.copy()
        if ttype == 'minmax':
            reduced = reduced * (maxi - mini)[:,None] + mini[:,None]
        elif ttype =='cuberoot':
            reduced = reduced**3

        reduced[np.isnan(reduced)] = 0; reduced[np.isinf(reduced)] = 0
        return reduced

