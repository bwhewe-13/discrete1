
import numpy as np
import numpy.ctypeslib as npct
import ctypes

def update_q(xs,phi,start,stop,g):
    return np.sum(xs[:,g,start:stop]*phi[:,start:stop],axis=1)

def surface_area_calc(rho):
    return 4 * np.pi * rho**2

def volume_calc(plus,minus):
    return 4 * np.pi / 3 * (plus**3 - minus**3)

def half_angle(psi_plus,total,delta,source):
    """ This is for finding the half angle (N = 1/2) at cell i """
    psi_nhalf = np.zeros((len(total)))
    for ii in range(len(total)-1,-1,-1):
        psi_nhalf[ii] = (2 * psi_plus + delta * source[ii] ) / \
            (2 + total[ii] * delta)
        psi_plus = 2 * psi_nhalf[ii] - psi_plus
    return psi_nhalf

def creating_weights(angles):
    mu, w = np.polynomial.legendre.leggauss(angles)
    w /= np.sum(w)
    return mu, w


# class CFunctions:

#     def __init__(self,groups,cells,materials):
#         self.groups = groups
#         self.cells = cells
#         self.materials = materials

#     class xs_vector:
#         _fields_ = [("array", (ctypes.c_double * self.materials) * self.groups)]

#     class xs_matrix:
#         _fields_ = [("array", ((ctypes.c_double * self.groups) * self.groups) * \
#                         self.materials)]

#     class boundary_edges:
#         _fields_ = [("array", (ctypes.c_double * 2) * self.groups)]

#     class spatial_energy:
#         _fields_ = [("array", (ctypes.c_double * self.cells) * self.groups)]

#     def c_sweeps(self):
#         ...
