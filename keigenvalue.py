""" Setting Up Criticality problems for critical.py """

from .generate import XSGenerate087, XSGenerate618

import numpy as np
import pkg_resources
import json

DATA_PATH = pkg_resources.resource_filename('discrete1','data/')

# class KEigenvalue:

#     def __init__(self,problem,enrich,orient='orig'):
#         self.problem = problem
#         self.enrich = enrich
#         self.orient = orient

#     @classmethod
#     def initialize(cls,problem,enrich,orient='orig'):
#         setup = cls(problem,enrich,orient)
#         dj_vars = {}
#         keys = 
#         if problem in ['hdpe','ss440']:
#             ss_vars = Problem1.steady(problem,enrich,orient)
#         elif problem in ['pu']:
#             ss_vars = Problem2.steady('hdpe',enrich,orient)


class Problem1:
    # 87 Group problem
    def orientation(refl,enrich,orient='orig'):
        if orient == 'orig':
            shape = [45,35,20]
            materials = [refl,['uh3',enrich],['uh3',0]]
        elif orient == 'multi':
            shape = [10]*8 + [20]
            materials = [refl,['uh3',enrich]] * 4 + [['uh3',0]]
        elif orient == 'mix1':
            shape = [45,5,25,5,20]
            materials = [refl,['uh3',0.12],['uh3',0.27],['uh3',0.12],['uh3',0]]
        return shape,materials

    def steady(refl,enrich,orient='orig'):
        shape,materials = Problem1.orientation(refl,enrich,orient)

        L = 0; R = sum(shape); I = 1000
        G = 87; N = 8
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w)
        # Take only half (reflective)
        N = int(0.5*N); mu = mu[N:]; w = w[N:]

        delta = R/I

        xs_total,xs_scatter,xs_fission = Tools.populate_xs_list(materials)
        layers = [int(ii/delta) for ii in shape]

        total_,scatter_,fission_ = Tools.populate_full_space(xs_total,xs_scatter,xs_fission,layers)

        return G,N,mu,w,total_,scatter_,fission_,I,delta

    def labeling(refl,enrich,orient='orig'):
        """ Returns the labels and splits of materials """
        shape,materials = Problem1.orientation(refl,enrich,orient)
        _,_,_,_,_,_,_,_,delta = Problem1.steady(refl,enrich,orient)
        labels, splits = Tools.create_slices(shape,materials,delta)
        return labels,splits

class Problem2:
    # 618 Group problem
    def orientation(refl,enrich,orient='orig'):
        if orient == 'orig':
            shape = [5,1.5,3.5]
            # Depleted is 0% Pu-239
            materials = [refl,['pu',1-enrich],['pu',0]]
        return shape,materials

    def steady(refl,enrich,orient='orig'):
        shape,materials = Problem2.orientation(refl,enrich,orient)

        L = 0; R = sum(shape); I = 1000
        G = 618; N = 8
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w)
        # Take only half (reflective)
        N = int(0.5*N); mu = mu[N:]; w = w[N:]

        delta = R/I

        xs_total,xs_scatter,xs_fission = XSGenerate618.cross_section(enrich)
        layers = [int(ii/delta) for ii in shape]

        total_,scatter_,fission_ = Tools.populate_full_space(xs_total,xs_scatter,xs_fission,layers)

        return G,N,mu,w,total_,scatter_,fission_,I,delta

    def labeling(refl,enrich,orient='orig'):
        """ Returns the labels and splits of materials """
        shape,materials = Problem2.orientation(refl,enrich,orient)
        _,_,_,_,_,_,_,_,delta = Problem2.steady(refl,enrich,orient)
        labels, splits = Tools.create_slices(shape,materials,delta)
        return labels,splits



class Tools:

    def populate_xs_list(materials):
        """ Populate list with cross sections of different materials """
        xs_total = []; xs_scatter = []; xs_fission = []
        # Iterate through materials list
        for mat in materials:
            # Check for Enrichment
            if type(mat) is list:
                iso = mat[0].upper()
                total_,scatter_,fission_ = XSGenerate087(iso,enrich=mat[1]).cross_section()
            else:
                total_,scatter_,fission_ = XSGenerate087(mat.upper()).cross_section()
            xs_total.append(total_); xs_scatter.append(scatter_); xs_fission.append(fission_)
            del total_, scatter_, fission_
        return xs_total, xs_scatter, xs_fission

    def populate_full_space(total,scatter,fission,layers):
        """ Populate lists into full space (I)
        total, scatter, fission: lists of cross sections of different materials
        layers: list of cell widths of each material
        """
        total_ = np.vstack([np.tile(total[count],(width,1)) for count,width in enumerate(layers)])
        scatter_ = np.vstack([np.tile(scatter[count],(width,1,1)) for count,width in enumerate(layers)])
        fission_ = np.vstack([np.tile(fission[count],(width,1,1)) for count,width in enumerate(layers)])

        return total_,scatter_,fission_

    def create_slices(shape,materials,delta):
        """ Create the labels and material splits for DJINN """
        initial = 0
        labels = []; splits = {}
        splits['refl'] = []; splits['fuel'] = []

        for sh,mat in zip(shape,materials):
            dist = int(sh/delta)
            if mat in _Constants.reflective_materials:
                splits['refl'].append(slice(initial,dist+initial))
                name = np.round(_Constants.compound_density[mat.upper()][0],2)
                labels += [name] * dist
            else:
                splits['fuel'].append(slice(initial,dist+initial))
                labels += [mat[1]] * dist
            
            initial += dist

        return np.array(labels),splits


class _Constants:
    reflective_materials = ['hdpe','ss440']
    # ['molar mass','density']
    compound_density = json.load(open(DATA_PATH + 'compound_density.json','r'))


