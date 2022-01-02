"""
Running Hybrid Problems (time dependent)
"""

from .fixed import FixedSource, ControlRod
from .fixed import Tools as Extra
from discrete1.utils import transport_tools as tools

import numpy as np
import ctypes
import pkg_resources

C_PATH = pkg_resources.resource_filename('discrete1','c/')

class Hybrid:
    def __init__(self,ptype,G,N):
        """ G and N are lists of [uncollided,collided]  """
        self.ptype = ptype
        self.geometry = 'slab'
        self.Gu,self.Gc = G if type(G) is list else [G,G]
        self.Nu,self.Nc = N if type(N) is list else [N,N]

    @classmethod
    def run(cls,ptype,G,N,T,dt,**kwargs):
        prob = cls(ptype,G,N)
        prob.T = T; prob.dt = dt
        prob.edges = kwargs['edges'] if 'edges' in kwargs else None
        prob.enrich = kwargs['enrich'] if 'enrich' in kwargs else None
        prob.geometry = kwargs['geometry'] if 'geometry' in kwargs else 'slab'
        if 'td' in kwargs:
            prob.td = kwargs['td']
            if prob.td == 'BE':
                return prob.backward_euler()
            elif prob.td == 'BDF2':
                return prob.bdf2()
        return prob.backward_euler()

    def backward_euler(self):
        # Set up Problem
        un_attr,un_keys = FixedSource.initialize(self.ptype,self.Gu,self.Nu,T=self.T,dt=self.dt,hybrid=self.Gu,enrich=self.enrich,edges=self.edges)
        uncollided = Uncollided(*un_attr)

        col_attr,col_keys = FixedSource.initialize(self.ptype,self.Gc,self.Nc,T=self.T,dt=self.dt,hybrid=self.Gu,enrich=self.enrich,edges=self.edges)
        collided = Collided(*col_attr)
        # Initialize Speeds
        speed_u = 1/(un_keys['v']*un_keys['dt'])
        speed_c = 1/(col_keys['v']*col_keys['dt'])

        # Initialize collided scalar flux
        phi_c = np.zeros((collided.I,collided.G)); time_phi = []
        psi_last = np.zeros((uncollided.I,uncollided.N,uncollided.G))
        
        # uncollided.total,uncollided.scatter,uncollided.fission = ControlRod.xs_update(uncollided.G,enrich=self.enrich,switch=0)
        # collided.total,collided.scatter,collided.fission = ControlRod.xs_update(collided.G,enrich=self.enrich,switch=0)

        if self.ptype in ['ControlRod']:
            scalar_flux = np.load('mydata/control_rod_critical/carbon_g87_phi_{}.npy'.format(str(int(self.enrich*100)).zfill(2)))
            source_q = np.einsum('ijk,ik->ij',uncollided.scatter,scalar_flux) + np.einsum('ijk,ik->ij',uncollided.fission,scalar_flux) 
            temp,psi_last = uncollided.multi_group(psi_last, speed_u, source=source_q, ts=0)
            del scalar_flux, temp, source_q
        # For calculating the number of time steps (computer rounding error)
        steps = int(np.round(un_keys['T']/un_keys['dt'],5))
        # Determining source for problem
        if self.ptype in ['Stainless','UraniumStainless','StainlessUranium']:
            # full_point_source = tools.discontinuous(uncollided.point_source,steps)
            full_point_source = tools.continuous(uncollided.point_source,steps)
        else:
            full_point_source = tools.stagnant(uncollided.point_source,steps)

        for t in range(steps):
            if self.ptype in ['ControlRod']:
                # The change of carbon --> stainless in problem
                switch = min(max(np.round(1 - 10**-(len(str(steps))-1)*10**(len(str(int(un_keys['T']/1E-6)))-1) * t,2),0),1)
                print('Switch {} Step {}'.format(switch,t))
                uncollided.total,uncollided.scatter,uncollided.fission = ControlRod.xs_update(uncollided.G,enrich=self.enrich,switch=switch)
                collided.total,collided.scatter,collided.fission = ControlRod.xs_update(collided.G,enrich=self.enrich,switch=switch)
            # Step 1: Solve Uncollided Equation
            phi_u,_ = uncollided.multi_group(psi_last,speed_u,self.geometry,ts=str(t)+'a')
            # Step 2: Compute Source for Collided
            source_c = np.einsum('ijk,ik->ij',uncollided.scatter,phi_u) + np.einsum('ijk,ik->ij',uncollided.fission,phi_u) 
            # Resizing
            if self.Gu != self.Gc:
                source_c = tools.big_2_small(source_c,un_keys['delta_e'],col_keys['delta_e'],col_keys['splits'])
            # Step 3: Solve Collided Equation
            phi_c = collided.multi_group(speed_c,source_c,phi_c,self.geometry,ts=t)
            # Resize phi_c
            if self.Gu != self.Gc:
                # phi = tools.small_2_big(phi_c,self.delta_u,self.delta_c,self.splits) + phi_u
                phi = tools.small_2_big(phi_c,un_keys['delta_e'],col_keys['delta_e'],col_keys['splits']) + phi_u
            else:
                phi = phi_c + phi_u
            # Step 4: Calculate next time step
            source = np.einsum('ijk,ik->ij',uncollided.scatter,phi) + \
                     uncollided.external_source + \
                     np.einsum('ijk,ik->ij',uncollided.fission,phi)
            phi,psi_next = uncollided.multi_group(psi_last,speed_u,self.geometry,source,ts=str(t)+'b')
            print('Time Step',t,'Flux',np.sum(phi),'\n===================================')
            # Step 5: Update and repeat
            psi_last = psi_next.copy()
            time_phi.append(phi)
            uncollided.point_source = full_point_source[t].copy()
        return phi,time_phi

    def bdf2(self):
        # Set up Problem
        un_attr,un_keys = FixedSource.initialize(self.ptype,self.Gu,self.Nu,T=self.T,dt=self.dt,hybrid=self.Gu,enrich=self.enrich,edges=self.edges)
        uncollided = Uncollided(*un_attr,td='BDF2')
        col_attr,col_keys = FixedSource.initialize(self.ptype,self.Gc,self.Nc,T=self.T,dt=self.dt,hybrid=self.Gu,enrich=self.enrich,edges=self.edges)
        collided = Collided(*col_attr,td='BDF2')        
        # Initialize speed
        speed_u = 1/(un_keys['v']*un_keys['dt'])
        speed_c = 1/(col_keys['v']*col_keys['dt'])
        # Initialize collided scalar flux
        phi_c = np.zeros((collided.I,collided.G)); time_phi = []
        psi_n0 = np.zeros((uncollided.I,uncollided.N,uncollided.G)); psi_n1 = psi_n0.copy()        
        # Initialize ControlRod problem differently
        if self.ptype in ['ControlRod']:
            # psi_n1 = np.tile(np.expand_dims(np.load('discrete1/data/initial_rod.npy'),axis=1),(1,uncollided.N,1))
            psi_last = np.tile(np.expand_dims(np.load('mydata/control_rod_critical/carbon_g87_phi_20.npy'),axis=1),(1,uncollided.N,1))
            collided.td = 'BE'; uncollided.td = 'BE'
        # For calculating the number of time steps (computer rounding error)
        steps = int(np.round(un_keys['T']/un_keys['dt'],5))
        # Determining source for problem
        if self.ptype in ['Stainless','UraniumStainless','StainlessUranium']:
            # full_point_source = tools.discontinuous(uncollided.point_source,steps)
            full_point_source = tools.continuous(uncollided.point_source,steps)
        else:
            full_point_source = tools.stagnant(uncollided.point_source,steps)
        for t in range(steps):
            if self.ptype in ['ControlRod']:
                # The change of carbon --> stainless in problem
                switch = min(max(np.round(1 - 10**-(len(str(steps))-1)*10**(len(str(int(un_keys['T']/1E-6)))-1) * t,2),0),1)
                #print('Switch {} Step {}'.format(switch,t))
                uncollided.total,uncollided.scatter,uncollided.fission = ControlRod.xs_update(uncollided.G,enrich=0.22,switch=switch)
                collided.total,collided.scatter,collided.fission = ControlRod.xs_update(collided.G,enrich=0.22,switch=switch)
            # Backward Euler for first step, BDF2 for rest
            psi_last = psi_n1.copy() if t == 0 else 2 * psi_n1 - 0.5 * psi_n0
            # Step 1: Solve Uncollided Equation
            phi_u,_ = uncollided.multi_group(psi_last,speed_u,self.geometry)
            # Step 2: Compute Source for Collided
            source_c = np.einsum('ijk,ik->ij',uncollided.scatter,phi_u) + np.einsum('ijk,ik->ij',uncollided.fission,phi_u)
            # Resizing
            if self.Gu != self.Gc:
                source_c = tools.big_2_small(source_c,un_keys['delta_e'],col_keys['delta_e'],col_keys['splits'])
            # Step 3: Solve Collided Equation
            phi_c = collided.multi_group(speed_c,source_c,phi_c,self.geometry)
            # Resize phi_c
            if self.Gu != self.Gc:
                # phi = tools.small_2_big(phi_c,self.delta_u,self.delta_c,self.splits) + phi_u
                phi = tools.small_2_big(phi_c,un_keys['delta_e'],col_keys['delta_e'],col_keys['splits']) + phi_u
            else:
                phi = phi_c + phi_u
            # Step 4: Calculate next time step
            source = np.einsum('ijk,ik->ij',uncollided.scatter,phi) + uncollided.source + np.einsum('ijk,ik->ij',uncollided.fission,phi)
            phi,psi_next = uncollided.multi_group(psi_last,speed_u,self.geometry,source)
            # Step 5: Update and repeat
            #print('Time Step',t,'Flux',np.sum(phi),'\n===================================')
            psi_n0 = psi_n1.copy()
            psi_n1 = psi_next.copy()
            time_phi.append(phi)
            uncollided.point_source = full_point_source[t].copy()
            # change to BDF2 time steps
            collided.td = 'BDF2'; uncollided.td = 'BDF2'
        return phi,time_phi


class Uncollided:
    def __init__(self, G, N, mu, w, total, scatter, fission, external_source, I,\
                    delta, point_source, td='BE'):
        self.G = G; self.N = N; 
        self.mu = mu; self.w = w
        self.total = total
        self.scatter = scatter
        self.fission = fission
        self.external_source = external_source
        self.I = I
        self.delta = 1/delta
        self.point_source_loc = point_source[0]
        self.point_source = point_source[1]
        self.td = td

    def slab(self, psi_last, speed, total_, source_, boundary):
        """ Step 1 of Hybrid
        Arguments:
            Different variables for collided and uncollided except I and inv_delta 
            psi_last: last time step, of size I x N
            speed: 1/(v*dt)   """
        clibrary = ctypes.cdll.LoadLibrary(C_PATH+'cHybrid.so')
        sweep = clibrary.uncollided

        phi = np.zeros((self.I),dtype='float64')
        psi_next = np.zeros(psi_last.shape,dtype='float64')

        weight = self.mu * self.delta
        point_source = ctypes.c_double(boundary)
        for n in range(self.N):
            # Determine the direction
            direction = ctypes.c_int(int(np.sign(self.mu[n])))
            weight = np.sign(self.mu[n]) * self.mu[n] * self.delta
            # Collecting Angle
            psi_angle = np.zeros((self.I),dtype='float64')
            psi_ptr = ctypes.c_void_p(psi_angle.ctypes.data)
            # Source Terms
            rhs = (source_ + psi_last[:,n] * speed).astype('float64')
            rhs_ptr = ctypes.c_void_p(rhs.ctypes.data)

            if self.td == 'BE':
                top_mult = (weight - 0.5 * total_ - 0.5 * speed).astype('float64')
                bottom_mult = (1/(weight + 0.5 * total_ + 0.5 * speed)).astype('float64')
            elif self.td == 'BDF2':
                top_mult = (weight - 0.5 * total_ - 0.75 * speed).astype('float64')
                bottom_mult = (1/(weight + 0.5 * total_ + 0.75 * speed)).astype('float64')
            top_ptr = ctypes.c_void_p(top_mult.ctypes.data)
            bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)
            phi_ptr = ctypes.c_void_p(phi.ctypes.data)
            sweep(phi_ptr, psi_ptr, rhs_ptr, top_ptr, bot_ptr, ctypes.c_double(self.w[n]), \
                  point_source, ctypes.c_int(self.point_source_loc), direction)
            psi_next[:,n] = psi_angle.copy()
        return phi,psi_next

    def sphere(self, psi_last, speed, total_, source_, boundary):
        clib = ctypes.cdll.LoadLibrary(C_PATH + 'cHybridSP.so')

        edges = np.cumsum(np.insert(np.ones((self.I))*1/self.delta,0,0))
        V = tools.volume_calc(edges[1:],edges[:self.I])
        if self.td == 'BE':
            v_total = (V * total_ + speed).astype('float64')
        elif self.td == 'BDF2':
            v_total = (V * total_ + 1.5 * speed).astype('float64')
        v_ptr = ctypes.c_void_p(v_total.ctypes.data)

        psi_next = np.zeros(psi_last.shape,dtype='float64')
        psi_centers = np.zeros((self.N),dtype='float64')

        mu_minus = -1
        angular = np.zeros((self.I),dtype='float64')
        an_ptr = ctypes.c_void_p(angular.ctypes.data)
        phi = np.zeros((self.I),dtype='float64')
        for n in range(self.N):
            mu_plus = mu_minus + 2 * self.w[n]
            tau = (self.mu[n] - mu_minus) / (mu_plus - mu_minus)
            if n == 0:
                alpha_minus = ctypes.c_double(0.)
                psi_nhalf = (tools.half_angle(boundary,total_,1/self.delta, \
                             source_)).astype('float64')
            if n == self.N - 1:
                alpha_plus = ctypes.c_double(0.)
            else:
                alpha_plus = ctypes.c_double(alpha_minus - self.mu[n] * self.w[n])
            Q = (V * (source_) + psi_last[:,n] * speed).astype('float64') 
            q_ptr = ctypes.c_void_p(Q.ctypes.data)
            psi_ptr = ctypes.c_void_p(psi_nhalf.ctypes.data)
            phi_ptr = ctypes.c_void_p(phi.ctypes.data)
            clib.sweep(an_ptr, phi_ptr, psi_ptr, q_ptr, v_ptr, ctypes.c_double(1/self.delta),\
                ctypes.c_double(self.w[n]), ctypes.c_double(self.mu[n]), alpha_plus,\
                alpha_minus, ctypes.c_double(boundary), ctypes.c_int(self.point_source_loc), ctypes.c_double(tau))
            # Update angular center corrections
            psi_centers[n] = angular[0]
            psi_next[:,n] = angular.copy()
            # Update angular difference coefficients
            alpha_minus = alpha_plus
            mu_minus = mu_plus
        return phi,psi_next

    def multi_group(self, psi_last, speed, geometry='slab', source=None, ts=None):
        phi_old = np.zeros((self.I,self.G))
        psi_next = np.zeros(psi_last.shape)

        geo = getattr(Uncollided,geometry)  # Get the specific sweep
        if source is None:
            current = self.external_source.copy()
        else:
            current = source.copy()
        tol = 1e-12; MAX_ITS = 100
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)
            for g in range(self.G):
                phi[:,g],psi_next[:,:,g] = geo(self, psi_last[:,:,g], speed[g], \
                            self.total[:,g], current[:,g], self.point_source[g])
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            if np.isnan(change) or np.isinf(change):
                change = 0.
            # file_name = 'testdata/slab_uranium_stainless_ts0100_be/hybrid_uncollided_ts{}_{}'.format(str(ts).zfill(4),str(count).zfill(3))
            # np.save(file_name,[['time',end - start],['change',np.linalg.norm(phi - phi_old)]])
            # print('Uncollided Change is',change,'\n===================================')
            count += 1
            converged = (change < tol) or (count >= MAX_ITS) 
            phi_old = phi.copy()
        return phi,psi_next


class Collided:
    def __init__(self, G, N, mu, w, total, scatter, fission,external_source, I,\
                 delta, point_source, td='BE'):
        self.G = G; self.N = N; 
        self.mu = mu; self.w = w
        self.total = total
        self.scatter = scatter
        self.fission = fission
        # self.external_source = external_source
        self.I = I
        self.delta = 1/delta
        # self.point_source = point_source
        self.td = td

    def slab(self, speed, total_, scatter_, source_, guess_):
        clibrary = ctypes.cdll.LoadLibrary(C_PATH + 'cHybrid.so')
        sweep = clibrary.collided
        source_ = source_.astype('float64')
        source_ptr = ctypes.c_void_p(source_.ctypes.data)
        
        phi_old = guess_.copy()

        tol = 1e-8; MAX_ITS = 100
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros((self.I),dtype='float64')
            for n in range(self.N):
                # Determine the direction
                direction = ctypes.c_int(int(np.sign(self.mu[n])))
                weight = np.sign(self.mu[n]) * self.mu[n] * self.delta

                if self.td == 'BE':
                    top_mult = (weight - 0.5 * total_ - 0.5 * speed).astype('float64')
                    bottom_mult = (1/(weight + 0.5 * total_ + 0.5 * speed)).astype('float64')
                elif self.td == 'BDF2':
                    top_mult = (weight - 0.5 * total_ - 0.75 * speed).astype('float64')
                    bottom_mult = (1/(weight + 0.5 * total_ + 0.75 * speed)).astype('float64')
                top_ptr = ctypes.c_void_p(top_mult.ctypes.data)
                bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)
                temp_scat = (scatter_ * phi_old).astype('float64')
                ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)
                phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                sweep(phi_ptr,ts_ptr,source_ptr,top_ptr,bot_ptr,ctypes.c_double(self.w[n]),direction)
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            if np.isnan(change) or np.isinf(change):
                change = 0.
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()

        return phi

    def sphere(self, speed, total_, scatter_, source_, guess_):
        clib = ctypes.cdll.LoadLibrary(C_PATH + 'cHybridSP.so')

        edges = np.cumsum(np.insert(np.ones((self.I))*1/self.delta,0,0))
        V = tools.volume_calc(edges[1:],edges[:self.I])
        if self.td == 'BE':
            v_total = (V * total_ + speed).astype('float64')
        elif self.td == 'BDF2':
            v_total = (V * total_ + 1.5 * speed).astype('float64')
        v_ptr = ctypes.c_void_p(v_total.ctypes.data)

        phi_old = guess_.copy()
        psi_centers = np.zeros((self.N),dtype='float64')

        tol=1e-12; MAX_ITS=100
        converged = 0; count = 1; 
        while not (converged):
            mu_minus = -1
            angular = np.zeros((self.I),dtype='float64')
            an_ptr = ctypes.c_void_p(angular.ctypes.data)
            phi = np.zeros((self.I),dtype='float64')
            for n in range(self.N):
                mu_plus = mu_minus + 2 * self.w[n]
                tau = (self.mu[n] - mu_minus) / (mu_plus - mu_minus)
                if n == 0:
                    alpha_minus = ctypes.c_double(0.)
                    psi_nhalf = (tools.half_angle(0,total_,1/self.delta, source_ + scatter_ * phi_old)).astype('float64')
                if n == self.N - 1:
                    alpha_plus = ctypes.c_double(0.)
                else:
                    alpha_plus = ctypes.c_double(alpha_minus - self.mu[n] * self.w[n])
                Q = (V * (source_ + scatter_ * phi_old)).astype('float64')
                q_ptr = ctypes.c_void_p(Q.ctypes.data)
                psi_ptr = ctypes.c_void_p(psi_nhalf.ctypes.data)
                phi_ptr = ctypes.c_void_p(phi.ctypes.data)
                clib.sweep(an_ptr, phi_ptr, psi_ptr, q_ptr, v_ptr, ctypes.c_double(1/self.delta),\
                    ctypes.c_double(self.w[n]), ctypes.c_double(self.mu[n]), alpha_plus,\
                    alpha_minus, ctypes.c_double(0.), ctypes.c_int(self.I), ctypes.c_double(tau))
                # Update angular center corrections
                psi_centers[n] = angular[0]
                # Update angular difference coefficients
                alpha_minus = alpha_plus
                mu_minus = mu_plus
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi

    def multi_group(self,speed,source,guess,geometry='slab',ts=None):
        assert(source.shape[1] == self.G), 'Wrong Number of Groups'
        phi_old = guess.copy()
        geo = getattr(Collided,geometry)  # Get the specific sweep
        tol = 1e-12; MAX_ITS = 100
        converged = 0; count = 1
        while not (converged):
            phi = np.zeros(phi_old.shape)
            for g in range(self.G):
                q_tilde = source[:,g] + tools.update_q(self.scatter,phi_old,g+1,self.G,g) + \
                          tools.update_q(self.fission,phi_old,g+1,self.G,g)
                if g != 0:
                    q_tilde += tools.update_q(self.scatter,phi,0,g,g) + \
                               tools.update_q(self.fission,phi,0,g,g)
                phi[:,g] = geo(self, speed[g], self.total[:,g], self.scatter[:,g,g] +\
                               self.fission[:,g,g], q_tilde, phi_old[:,g])
            change = np.linalg.norm((phi - phi_old)/phi/(self.I))
            converged = (change < tol) or (count >= MAX_ITS) 
            count += 1
            phi_old = phi.copy()
        return phi