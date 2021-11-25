
import numpy as np
# import numpy.ctypeslib as npct
import ctypes
from scipy.special import erfc

def update_q(xs, phi, start, stop, g):
    return np.sum(xs[:,g,start:stop]*phi[:,start:stop],axis=1)

def surface_area_calc(rho):
    return 4 * np.pi * rho**2

def volume_calc(plus, minus):
    return 4 * np.pi / 3 * (plus**3 - minus**3)

def half_angle(psi_plus, total, delta, source):
    """ This is for finding the half angle (N = 1/2) at cell i """
    psi_nhalf = np.zeros((len(total)))
    for ii in range(len(total)-1,-1,-1):
        psi_nhalf[ii] = (2 * psi_plus + delta * source[ii] ) / \
            (2 + total[ii] * delta)
        psi_plus = 2 * psi_nhalf[ii] - psi_plus
    return psi_nhalf

def creating_weights(angles, boundary=[0,0]):
    mu, w = np.polynomial.legendre.leggauss(angles)
    w /= np.sum(w)
    if np.sum(boundary) == 1:
        angles = int(0.5 * angles)
        mu = mu[angles:]
        w = w[angles:]
    return angles, mu, w


############################
# Sources
############################
def stagnant(source,steps):
    return np.tile(source,(steps,1))

def continuous(source,steps):
    func = lambda a,b: list(erfc(np.arange(1,4))*a+b)
    full = np.zeros((steps,len(source)))
    group = np.argwhere(source != 0)[0,0]
    source = source[group]
    for t in range(steps):
        if t < int(0.2*steps):
            source *= 1
            full[t,group] = source
        elif t % int(0.1*steps) == 0:
            temp = t
            full[t:t+3,group] = func(source,0.5*source)
            source *= 0.5
        elif t in np.arange(temp+1,temp+3):
            continue
        else:
            full[t,group] = source
    return full

def discontinuous(source,steps):
    full = np.zeros((steps,len(source)))
    group = np.argwhere(source != 0)[0,0]
    source = source[group]
    for t in range(steps):
        if t < int(0.2*steps):
            source *= 1
        elif t % int(0.1*steps) == 0:
            source *= 0.5
        full[t,group] = source
    return full


############################
# Collapsing Groups
############################
def energy_distribution(big,small):
    """ List of slices for different energy sizes
    Arguments:
        big: uncollided energy groups, int
        small: collided energy groups, int
    Returns:
        list of slices   """
    new_grid = np.ones((small)) * int(big/small)
    new_grid[np.linspace(0,small-1,big % small,dtype=int)] += 1
    inds = np.cumsum(np.insert(new_grid,0,0),dtype=int)
    splits = [slice(ii,jj) for ii,jj in zip(inds[:small],inds[1:])]
    return splits

def small_2_big(mult_c,delta_u,delta_c,splits):
    Gu = len(delta_u)
    size = (mult_c.shape[0],Gu)
    mult_u = np.zeros(size)
    factor = delta_u.copy()
    for count,index in enumerate(splits):
        for ii in np.arange(index.indices(Gu)[0],index.indices(Gu)[1]):
            mult_u[:,ii] = mult_c[:,count]
            factor[ii] /= delta_c[count]
    mult_u *= factor
    return mult_u

def big_2_small(mult_u,delta_u,delta_c,splits):
    # size = (mult_u.shape[0],len(delta_c))
    size = mult_u.shape[:len(mult_u.shape)-1] + np.array(delta_c).shape
    mult_c = np.zeros(size)
    # Have to change this
    for count,index in enumerate(splits):
        # mult_c[:,:,count] = np.sum(mult_u[:,:,index],axis=2) 
        mult_c[:,count] = np.sum(mult_u[:,index],axis=1) 
    return mult_c
    
def big_2_small2(mult_u,delta_u,delta_c,splits):
    # size = (mult_u.shape[0],len(delta_c))
    size = mult_u.shape[:len(mult_u.shape)-1] + np.array(delta_c).shape
    mult_c = np.zeros(size)
    # Have to change this
    for count,index in enumerate(splits):
        mult_c[:,:,count] = np.sum(mult_u[:,:,index],axis=2) 
        # mult_c[:,count] = np.sum(mult_u[:,index],axis=1) 
    return mult_c