#!/usr/bin/env python

import numpy as np
from discrete1.util import chem,sn_tools
import discrete1.slab as s
import argparse

parser = argparse.ArgumentParser(description='Finding Critical Layers')
parser.add_argument('-r',action='store',dest='distance',nargs='+')
parser.add_argument('-d',action='store',dest='delta',type=float)
usr_input = parser.parse_args()

distance = [float(ii) for ii in usr_input.distance]
distance = distance + distance[1::-1]

# Layer densities
conc = 0.2; density_uh3 = 10.95; density_ch3 = 0.97
uh3_density = chem.density_list('UH3',density_uh3,conc)
hdpe_density = chem.density_list('CH3',density_ch3)
uh3_238_density = chem.density_list('U^238H3',density_uh3)

# Loading Cross Section Data
dim = 87; spec_temp = '00'
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
hdpe_fission = np.zeros((dim,dim))

# Cross section layers
xs_scatter = [hdpe_scatter.T,uh3_scatter.T,uh3_238_scatter.T,uh3_scatter.T,hdpe_scatter.T]
xs_total = [hdpe_total,uh3_total,uh3_238_total,uh3_total,hdpe_total]
xs_fission = [hdpe_fission.T,uh3_fission.T,uh3_238_fission.T,uh3_fission.T,hdpe_fission.T]

# Setting up eigenvalue equation
N = 8; L = 0; R = sum(distance)
mu,w = np.polynomial.legendre.leggauss(N)
w /= np.sum(w)
layers = [int(ii/usr_input.delta) for ii in distance]
I = int(sum(layers))

scatter_ = sn_tools.mixed_propagate(xs_scatter,layers,G=dim,L=L,dtype='scatter')
fission_ = sn_tools.mixed_propagate(xs_fission,layers,G=dim,dtype='fission2')
total_ = sn_tools.mixed_propagate(xs_total,layers,G=dim)

problem = s.eigen(dim,N,mu,w,total_,scatter_,fission_,L,R,I)
phi,keff = problem.transport(LOUD=True)

# Track keff and distaces
outF = open("hist_layer.txt", "a")
line = 'Layers: {}     Delta: {}     Keff: {}'.format(str(distance),usr_input.delta,keff)
outF.write(line)
outF.write("\n")
outF.close()


