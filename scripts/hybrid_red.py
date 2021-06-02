
from discrete1.hybrid import Hybrid
from discrete1.source import Source

import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description='picking models')
parser.add_argument('-problem',action='store',dest='problem')
parser.add_argument('-angles',action='store',dest='angles',nargs='+')  # First is uncollided (larger)
parser.add_argument('-groups',action='store',dest='groups',nargs='+')  # First is uncollided (larger)
parser.add_argument('-steps',action='store',dest='steps')
parser.add_argument('-geometry',action='store',dest='geometry')
parser.add_argument('-time',action='store',dest='time')                # either BE or BDF2
parser.add_argument('-clock',action='store',dest='clock')
usr_input = parser.parse_args()

print('\nProblem {}\nAngles {}\nSteps {}\nGeometry {}\nTime {}'.format(usr_input.problem,
    usr_input.angles,usr_input.steps,usr_input.geometry,usr_input.time))

# Pick the problem to run
if usr_input.problem == 'uranium':
    label = 'uranium_stainless'
    if usr_input.geometry == 'sphere':
        part = 'StainlessUranium'
    elif usr_input.geometry == 'slab':
        part = 'UraniumStainless'
elif usr_input.problem == 'stainless':
    label = 'stainless'
    part = 'Stainless'
elif usr_input.problem == 'control':
    label = 'control_rod'
    part = 'ControlRod'

# Starting Time 
if usr_input.problem in ['stainless','uranium']:
    T = 10E-7
elif usr_input.problem in ['control']:
    T = 25E-6

dt = np.round(T/int(usr_input.steps),10)
Ts = str(usr_input.steps).zfill(4)
print('dt {}\nTime Label {}'.format(dt,Ts))

# Angles
angle1 = int(usr_input.angles[0])
an1 = str(angle1).zfill(2)
angle2 = int(usr_input.angles[1])
an2 = str(angle2).zfill(2)

if usr_input.groups is None:
    groups = [87,80,60,43,21,10]
else:
    groups = [int(ii) for ii in usr_input.groups]

# Iterate over collided groups
for gg in groups:
    print('\nEnergy Groups {}\n================='.format(gg))
    timer = []
    if label == 'uranium_stainless':
        start = time.time()
        phi,full = Hybrid.run(part,[87,gg],[angle1,angle2],T=T,dt=dt,enrich=0.2,geometry=usr_input.geometry,td=usr_input.time)
        end = time.time()
    elif label == 'stainless':
        start = time.time()
        phi,full = Hybrid.run(part,[87,gg],[angl1,angle2],T=T,dt=dt,geometry=usr_input.geometry,td=usr_input.time)
        end = time.time()
    elif label == 'control_rod':
        start = time.time()
        phi,full = Hybrid.run(part,[87,gg],[angle1,angle2],T=T,dt=dt,enrich=0.22,geometry=usr_input.geometry,td=usr_input.time)
        end = time.time()
    timer.append(end-start)
    # np.save('mydata/{}_{}_ts{}_{}/hybrid_g87_g{}_s{}_s{}_auto'.format(usr_input.geometry,label,Ts,usr_input.time.lower(),gg,an1,an2),phi)

    if usr_input.clock is None:
        mydict = {}
        for ts in range(len(full)):
            lab = 'ts{}'.format(str(ts).zfill(4))
            mydict[lab] = full[ts]
        np.savez('mydata/{}_{}_ts{}_{}/full_hybrid_g87_g{}_s{}_s{}_auto'.format(usr_input.geometry,label,Ts,usr_input.time.lower(),gg,an1,an2),**mydict)
        del full,mydict,phi

    if usr_input.clock is not None:
        for ii in range(4):
            if label == 'uranium_stainless':
                start = time.time()
                _,_ = Hybrid.run(part,[87,gg],[angle1,angle2],T=T,dt=dt,enrich=0.2,geometry=usr_input.geometry,td=usr_input.time)
                end = time.time()
            elif label == 'stainless':
                start = time.time()
                _,_ = Hybrid.run(part,[87,gg],[angle1,angle2],T=T,dt=dt,geometry=usr_input.geometry,td=usr_input.time)
                end = time.time()
            elif label == 'control_rod':
                start = time.time()
                _,_ = Hybrid.run(part,[87,gg],[angle1,angle2],T=T,dt=dt,enrich=0.22,geometry=usr_input.geometry,td=usr_input.time)
                end = time.time()
            timer.append(end-start)
        np.save('mydata/{}_{}_ts{}_{}/hybrid_g87_g{}_s{}_s{}_auto'.format(usr_input.geometry,label,Ts,usr_input.time.lower(),gg,an1,an2),timer)
        del timer

