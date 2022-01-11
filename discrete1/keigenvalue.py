""" Setting Up Criticality problems for critical.py """

from .generate import XSGenerate087, XSGenerate618
from .fixed import Tools as ReduceTools

import numpy as np
import pkg_resources
import json
import os

DATA_PATH = pkg_resources.resource_filename('discrete1','data/')
C_PATH = pkg_resources.resource_filename('discrete1','c/')

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
        # N = int(0.5*N); mu = mu[N:]; w = w[N:]
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
            # materials = [refl,['pu',1-enrich],['pu',0]]
        # Perturbations
        elif orient == 'dep_enr_01':  # Deplete + 0.5 / Enrich - 0.5
            shape = [5,1,4]
        elif orient == 'dep_enr_02':  # Deplete - 0.5 / Enrich + 0.5
            shape = [5,2,3]

        elif orient == 'enr_refl_01':  # Enrich + 0.5 / Refl - 0.5
            shape = [4.5,2,3.5]
        elif orient == 'enr_refl_02':  # Enrich - 0.5 / Refl + 0.5
            shape = [5.5,1,3.5]

        elif orient == 'refl_dep_01':  # Refl + 0.5 / Deplete - 0.5
            shape = [5.5,1.5,3]
        elif orient == 'refl_dep_02':  # Refl - 0.5 / Deplete + 0.5
            shape = [4.5,1.5,4]

        # Have to work on these
        elif orient == 'enr_refl_03':  # Enrich - 0.2 / Refl + 0.2
            shape = [5.2,1.3,3.5]
        elif orient == 'dep_enr_03':   # Deplete + 0.2 / Enrich - 0.2
            shape = [5,1.3,3.7]

        elif orient == 'enr_refl_04':  # Enrich - 0.4 / Refl + 0.4
            shape = [5.4,1.1,3.5]
        elif orient == 'dep_enr_04':   # Deplete + 0.4 / Enrich - 0.4
            shape = [5,1.1,3.9]

        # Depleted is 0% Pu-239
        materials = [refl,['pu',1-enrich],['pu',0]]
        return shape,materials

    def steady(refl,enrich,orient='orig',groups=618,naive=False):
        shape,materials = Problem2.orientation(refl,enrich,orient)

        L = 0; R = sum(shape); I = 1000
        G = groups; N = 16
        reduced = True if G != 87 else False

        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w)
        # Take only half (reflective)
        N = int(0.5*N); mu = mu[N:]; w = w[N:]
        Tools.recompile(I,N)

        delta = R/I
        # xs_total,xs_scatter,xs_fission = XSGenerate618.cross_section(enrich)
        layers = [int(ii/delta) for ii in shape]

        if reduced and naive == 'naive':
            xs_total,xs_scatter,xs_fission = XSGenerate618.cross_section_reduce(G,enrich)
        else:
            xs_total,xs_scatter,xs_fission = XSGenerate618.cross_section(enrich)

        total_,scatter_,fission_ = Tools.populate_full_space(xs_total,xs_scatter,xs_fission,layers)
        if reduced and naive == 'known':
            ref_flux = np.load('discrete1/data/pluto_phi_{}.npy'.format(str(int(enrich*100)).zfill(2)))
            grid = np.load('discrete1/data/energy_edges_618G.npy')
            total_, scatter_, fission_ = Tools.group_reduction_flux(G, ref_flux, grid, \
                                              total_, scatter_, fission_)
        return G,N,mu,w,total_,scatter_,fission_,I,delta

    def labeling(refl,enrich,orient='orig'):
        """ Returns the labels and splits of materials """
        shape,materials = Problem2.orientation(refl,enrich,orient)
        _,_,_,_,_,_,_,_,delta = Problem2.steady(refl,enrich,orient)
        labels, splits = Tools.create_slices(shape,materials,delta)
        return labels,splits

class Problem3:
    # 87 Group problem
    def orientation(refl,enrich,orient='c'):
        # if orient == 'orig':  # carbon in middle
        #     # shape = [10,4,6]
        #     # materials = [refl,['u',enrich],refl]
        #     shape = [10,4,12,4,10]
        #     materials = [refl,['u',enrich],orient,['u',enrich],refl]
        shape = [10,4,12,4,10]
        materials = ['c',['u',enrich],refl,['u',enrich],'c']
        return shape,materials

    def steady(refl,enrich,orient='c',groups=87):
        shape,materials = Problem3.orientation(refl,enrich,orient)

        L = 0; R = sum(shape); I = 1000
        G = groups; N = 8
        reduced = True if G != 87 else False

        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w)
        # Take only half (reflective)
        # N = int(0.5*N); mu = mu[N:]; w = w[N:]

        delta = R/I

        xs_total,xs_scatter,xs_fission = Tools.populate_xs_list(materials)
        layers = [int(ii/delta) for ii in shape]

        if reduced:
            energy_grid = np.load(DATA_PATH + 'energyGrid.npy')
            edges = None
            xs_total,xs_scatter,xs_fission = ReduceTools.group_reduction(G,energy_grid,xs_total,xs_scatter,xs_fission,edges)
        total_,scatter_,fission_ = Tools.populate_full_space(xs_total,xs_scatter,xs_fission,layers)
        return G,N,mu,w,total_,scatter_,fission_,I,delta

    def labeling(refl,enrich,orient='orig'):
        """ Returns the labels and splits of materials """
        shape,materials = Problem3.orientation(refl,enrich,orient)
        _,_,_,_,_,_,_,_,delta = Problem3.steady(refl,enrich,orient)
        labels, splits = Tools.create_slices(shape,materials,delta)
        return labels,splits

class Problem4:
    # 87 Group problem
    def orientation(refl,enrich,orient='orig'):
        if orient == 'uranium':
            # shape = [10,5,5]
            # materials = [refl,['u',enrich],['u',0]]
            # shape = [10]
            # materials = [['u',enrich]]
            shape = [5,5,10]
            materials = [['u',0],['u',enrich],refl]
        elif orient == 'hydride':
            # shape = [10,5,5]
            # materials = [refl,['uh3',enrich],['uh3',0]]
            shape = [5,5,10]
            materials = [['uh3',0],['uh3',enrich],refl]
        elif orient == 'mix1':
            shape = [5,5,5,10]
            materials = [['u',0],['u',0.5],['u',1.0],refl]
        elif orient == 'mix2':
            shape = [5,5,5,10]
            materials = [['u',0],['u',0.3],['u',0.6],refl]
        elif orient == 'mix3':
            shape = [5,5,5,10]
            materials = [['uh3',0],['uh3',0.5],['uh3',1.0],refl]
        elif orient == 'mix4':
            shape = [5,5,5,10]
            materials = [['uh3',0],['uh3',0.3],['uh3',0.6],refl]

        return shape,materials

    def steady(refl,enrich,orient='orig',sn=8):
        shape,materials = Problem4.orientation(refl,enrich,orient)
        L = 0; R = sum(shape); I = 1000
        G = 87; N = sn
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w)
        # Take only half (reflective)
        # N = int(0.5*N); mu = mu[N:]; w = w[N:]
        delta = R/I
        xs_total,xs_scatter,xs_fission = Tools.populate_xs_list(materials)
        layers = [int(ii/delta) for ii in shape]
        assert sum(layers) == I
        total_,scatter_,fission_ = Tools.populate_full_space(xs_total,xs_scatter,xs_fission,layers)
        return G,N,mu,w,total_,scatter_,fission_,I,delta

    def labeling(refl,enrich,orient='orig'):
        """ Returns the labels and splits of materials """
        shape,materials = Problem4.orientation(refl,enrich,orient)
        _,_,_,_,_,_,_,_,delta = Problem4.steady(refl,enrich,orient)
        labels, splits = Tools.create_slices(shape,materials,delta)
        return labels,splits


class Tools:

    def index_generator(big,small):
        new_grid = np.ones((small)) * int(big / small)
        new_grid[np.linspace(0,small-1,big % small,dtype=int)] += 1
        assert (new_grid.sum() == big)
        return np.cumsum(np.insert(new_grid,0,0),dtype=int)

    def vector_reduction(vector,idx):
        new_group = len(idx) - 1
        if len(vector.shape) == 2:
            return np.array([np.sum(vector[:,idx[ii]:idx[ii+1]],axis=1) for ii in range(new_group)])
        return np.array([sum(vector[idx[ii]:idx[ii+1]]) for ii in range(new_group)])

    def group_reduction_flux(new_group,flux,grid,xs_total,xs_scatter,xs_fission,inds=None):
        I = xs_total.shape[0]
        inds = Tools.index_generator(len(flux[0])-1,new_group) if inds is None else inds
        diff_grid = np.diff(grid)
        
        temp_total = Tools.vector_reduction(xs_total.copy()*diff_grid*flux,inds)
        new_total = (temp_total / (Tools.vector_reduction(flux*diff_grid,inds))).T
        
        xs_scatter = np.einsum('ijk,ik->ijk',xs_scatter.copy(), flux*diff_grid)
        xs_fission = np.einsum('ijk,ik->ijk',xs_fission.copy(), flux*diff_grid)
        
        new_scatter = np.zeros((I,new_group,new_group))
        new_fission = np.zeros((I,new_group,new_group))
        
        for ii in range(new_group):
            idx1 = slice(inds[ii],inds[ii+1])
            for jj in range(new_group):
                idx2 = slice(inds[jj],inds[jj+1])
                new_scatter[:,ii,jj] = np.sum(xs_scatter[:,idx1,idx2],axis=(2,1)) / np.sum(flux[:,idx2]*diff_grid[idx2],axis=1)
                new_fission[:,ii,jj] = np.sum(xs_fission[:,idx1,idx2],axis=(2,1)) / np.sum(flux[:,idx2]*diff_grid[idx2],axis=1)
        return new_total, new_scatter, new_fission

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

    def recompile(I,N):
        # Recompile cCritical
        command = f'gcc -fPIC -shared -o {C_PATH}cCritical.so {C_PATH}cCritical.c -DLENGTH={I}'
        os.system(command)
        # Recompile cCriticalSP
        command = f'gcc -fPIC -shared -o {C_PATH}cCriticalSP.so {C_PATH}cCriticalSP.c -DLENGTH={I} -DN={N}'
        os.system(command)


class _Constants:
    reflective_materials = ['hdpe','ss440']
    # ['molar mass','density']
    compound_density = json.load(open(DATA_PATH + 'compound_density.json','r'))

