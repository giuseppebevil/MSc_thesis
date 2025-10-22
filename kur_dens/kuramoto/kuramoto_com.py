import numpy as np
from scipy.integrate import odeint, solve_ivp
import scipy as sci
import random
import powerlaw
from datetime import datetime
from numba import jit
from jitcode import jitcode, y
import time
from scipy.stats import cauchy
from scipy.stats import rv_continuous

@jit(nopython=True)
def phase_coherence_numba(angles_vec):
    '''
    Compute global order parameter R_t - mean length of resultant vector
    '''
    suma = 0
    for i in angles_vec:
        suma += (np.exp(1j*i))
    r = np.abs(suma / len(angles_vec))

    return r

@jit(nopython=True)
def derivative_numba_com(init, t, adj_mat1, adj_mat2, coupling1, coupling2, natfreqs1, natfreqs2, omega_f, A, B):    
    '''
    Modified version with the external field
    Compute derivative of all nodes for current state, defined as

        dx_i,1      natfreq_i,1 +  k1  sum_j,1 ( Aij,1 * sin (angle_j,1 - angle_i,1) ) + h * sin(omega_f1*t - angle_i,1)
       -------   =                ----
        dt                       M_i,1

        dx_i,2      natfreq_i,2 +  k2  sum_j,2 ( Aij,2 * sin (angle_j,2 - angle_i,2) ) + h * sin(omega_f2*t - angle_i,2)
       -------   =                 ----
        dt                        M_i,2

        dh      f(x_i,1, x_i,2)
       ---- =             
        dt                

    '''
    n_nodes1 = len(adj_mat1)
    n_nodes2 = len(adj_mat2)
    angles_vec1 = init[:n_nodes1]
    angles_vec2 = init[n_nodes1:n_nodes1+n_nodes2]
    ext = init[-1]
    # r1 = init[-4]
    # r2 = init[-3]
    # psi1 = init[-2]
    # psi2 = init[-1]
    assert len(angles_vec1) == len(natfreqs1) == len(adj_mat1), \
        'Input dimensions for the first matrix do not match, check lengths'
        
    assert len(angles_vec2) == len(natfreqs2) == len(adj_mat2), \
        'Input dimensions for the second matrix do not match, check lengths'
        
    if np.abs(ext) >= 20:
        a = 0.0 * np.ones(n_nodes1)
        b = 0.0 * np.ones(n_nodes2)
        c = 0.0 * np.ones(1)
        d = np.concatenate((a, b, c))
        return d
    else:
        interactions1 = np.zeros(n_nodes1)  # Initialize interaction array
        interactions2 = np.zeros(n_nodes2)
        
        for i in range(n_nodes1):
            for j in range(n_nodes1):
                interactions1[i] += coupling1/n_nodes1 * adj_mat1[i, j] * np.sin(angles_vec1[j] - angles_vec1[i])
        for i in range(n_nodes2):
            for j in range(n_nodes2):
                interactions2[i] += coupling2/n_nodes2 * adj_mat2[i, j] * np.sin(angles_vec2[j] - angles_vec2[i])

        interactions_ext1 = ext * np.sin(omega_f * t - angles_vec1)
        interactions_ext2 = ext * np.sin(omega_f * t - angles_vec2)
        dx1dt = natfreqs1 + interactions1 + interactions_ext1  # sum over incoming interactions (over columns)
        dx2dt = natfreqs2 + interactions2 + interactions_ext2  # sum over incoming interactions (over columns)
        dbdt1 = 0.0
        dbdt2 = 0.0
        dbdt3 = 0.0
        for i in angles_vec1:
            dbdt1 += (np.exp(1j*i) / n_nodes1)
            
        for i in angles_vec2:
            dbdt2 += (np.exp(1j*i) / n_nodes2)

        for i in angles_vec1:
            for j in angles_vec2:
                dbdt3 += (np.exp(1j*(i-j))/(n_nodes1 * n_nodes2))
        dbdt1 = np.abs(dbdt1)
        dbdt2 = np.abs(dbdt2)
        dbdt3 = np.abs(dbdt3)

        dbdt = A*(dbdt1 + dbdt2)/2 - B*dbdt3
        dbdt = np.array([(dbdt)], dtype = 'float32')

        # dr1dt = 0.5 * coupling1 * 12 * r1 * (1 - np.power(r1, 2)) - delta * r1 + 0.5 * ext * (1 - np.power(r1, 2)) * np.cos(psi1)
        # dr2dt = 0.5 * coupling2 * 12 * r2 * (1 - np.power(r2, 2)) - delta * r2 + 0.5 * ext * (1 - np.power(r2, 2)) * np.cos(psi2)
        # dpsi1dt = -(omega_f1 + 0.5 * ext * (r1 + 1/r1) * np.sin(psi1))
        # dpsi2dt = -(omega_f2 + 0.5 * ext * (r2 + 1/r2) * np.sin(psi2))
        # dr1dt = np.array([(dr1dt)], dtype = 'float32')
        # dr2dt = np.array([(dr2dt)], dtype = 'float32')
        # dpsi1dt = np.array([(dpsi1dt)], dtype = 'float32')
        # dpsi2dt = np.array([(dpsi2dt)], dtype = 'float32')

        return np.concatenate((dx1dt, dx2dt, dbdt))
        #return np.concatenate((dx1dt, dx2dt, dbdt, dr1dt, dr2dt, dpsi1dt, dpsi2dt))

class distr(rv_continuous):
    def __init__(self, n, delta, **kwargs):
        self.n = n
        self.delta = delta
        super().__init__(**kwargs)
    def _pdf(self, omega):
        n = self.n
        delta = self.delta
        
        return n * np.sin(np.pi/(2*n))*delta**(2*n - 1)/(np.pi*(omega**(2*n) + delta**(2*n)))

class Kuramoto_com:

    def __init__(self, coupling1=1.0, coupling2 = 1.0, dt=0.01, t_max=10, n_nodes1=None, natfreqs1=None,
                 n_nodes2=None, natfreqs2=None, omega_f = 1.0, A = 1, B = 5, n = 4, delta = 1):
        '''
        coupling1 and coupling2: float
            Coupling strength. Default = 1. Typical values range between 0.4-2
        dt: float
            Delta t for integration of equations.
        t_max: float
            Total time of simulated activity.
            From that the number of integration steps is t_max/dt.
        n_nodes1 and n_nodes2: int, optional
            Number of oscillators.
            If None, it is inferred from len of natfreqs.
            Must be specified if natfreqs is not given.
        natfreqs1 and natfreqs2: 1D ndarray, optional
            Natural oscillation frequencies.
            If None, then new random values will be generated and kept fixed
            for the object instance.
            Must be specified if n_nodes is not given.
            If given, it overrides the n_nodes argument.
        '''
        if n_nodes1 is None and natfreqs1 is None:
            raise ValueError("n_nodes1 or natfreqs1 must be specified")
        if n_nodes2 is None and natfreqs2 is None:
            raise ValueError("n_nodes2 or natfreqs2 must be specified")
        self.A = A
        self.B = B
        self.dt = dt
        self.t_max = t_max
        self.coupling1 = coupling1
        self.coupling2 = coupling2
        self.omega_f = omega_f
        self.n = n
        self.delta = delta
        sample = distr(n = self.n, delta = self.delta)
        
        if natfreqs1 is not None:
            self.natfreqs1 = natfreqs1
            self.n_nodes1 = len(natfreqs1)
        else:
            #np.random.seed(89829)
            self.n_nodes1 = n_nodes1
            #self.natfreqs1 = cauchy.rvs(size = n_nodes1, loc = 0, scale = self.delta).astype('float32')
            #self.natfreqs1 = np.random.uniform(size = self.n_nodes1, low = -0.5, high = 0.5).astype('float32')
            self.natfreqs1 = np.random.normal(size = self.n_nodes1, loc = 0)
            #self.natfreqs1 = sample.rvs(size = n_nodes1)
        if natfreqs2 is not None:
            self.natfreqs2 = natfreqs2
            self.n_nodes2 = len(natfreqs2)
        else:
            #np.random.seed(223722)
            self.n_nodes2 = n_nodes2
            #self.natfreqs2 = cauchy.rvs(size = n_nodes2, loc = 0, scale = self.delta).astype('float32')
            #self.natfreqs2 = np.random.uniform(size = self.n_nodes2, low = -0.5, high = 0.5).astype('float32')
            self.natfreqs2 = np.random.normal(size = self.n_nodes2, loc = 0)
            #self.natfreqs2 = sample.rvs(size = n_nodes2)

    def init_angles(self, n_nodes):
        
        return 2 * np.pi * np.random.random(size = n_nodes).astype('float32')
    
    def derivative(self, init, t, adj_mat1, adj_mat2, coupling1, coupling2):
        # start = time.time()
        res = derivative_numba_com(init, t, adj_mat1, adj_mat2, coupling1, coupling2, self.natfreqs1, self.natfreqs2,
                                   self.omega_f, self.A, self.B
        )
        # end = time.time()
        # print(f"elapsed time ={end-start}")
        
        return res
        
    def integrate(self, angles_vec1, angles_vec2, adj_mat1, adj_mat2):
        '''Updates all states by integrating state of all nodes'''
        # Coupling term (k / Mj) is constant in the integrated time window.
        # Compute it only once here and pass it to the derivative function
        
        coupling1 = self.coupling1
        coupling2 = self.coupling2
        r1 = phase_coherence_numba(angles_vec1)
        r2 = phase_coherence_numba(angles_vec2)

        b_0 = 0
        b_0 = np.array([(b_0)], dtype = 'float32') # initial condition for the coupling of the field
        
        # psi1_0 = 0
        # psi2_0 = 0
        # r1 = np.array([(r1)], dtype = 'float32')
        # r2 = np.array([(r2)], dtype = 'float32')
        # psi1_0 = np.array([(psi1_0)], dtype = 'float32')
        # psi2_0 = np.array([(psi2_0)], dtype = 'float32')
        
        t = np.linspace(0, self.t_max, int(self.t_max/self.dt))
        init = np.concatenate((angles_vec1, angles_vec2, b_0))
        # init = np.concatenate((angles_vec1, angles_vec2, b_0, r1, r2, psi1_0, psi2_0))
        
        # duration = []
        # check = True #if both graphs are complete i use a faster expression to integrate
        # for i in range(self.n_nodes1):
        #     for j in range(self.n_nodes1):
        #         if j != i and adj_mat2[i,j]!=1:
        #             check = False
        # for i in range(self.n_nodes2):
        #     for j in range(self.n_nodes2):
        #         if j != i and adj_mat2[i,j]!=1:
        #             check = False
        # if check == True:
        #     print('Complete graphs')
        # else:
        #     print('Not complete graphs')
        timeseries = odeint(self.derivative, init, t, args=(adj_mat1, adj_mat2, coupling1, coupling2))
            
        return timeseries.T  # transpose for consistency (act_mat:node vs time)

    def run(self, adj_mat1=None, adj_mat2=None, angles_vec1=None, angles_vec2=None):
        '''
        adj_mat: 2D nd array
            Adjacency matrix representing connectivity.
        angles_vec: 1D ndarray, optional
            States vector of nodes representing the position in radians.
            If not specified, random initialization [0, 2pi].

        Returns
        -------
        act_mat: 2D ndarray
            Activity matrix: node vs time matrix with the time series of all
            the nodes.
        '''
        if angles_vec1 is None:
            #np.random.seed(58552)
            angles_vec1 = self.init_angles(self.n_nodes1)
        if angles_vec2 is None:
            #np.random.seed(123465)
            angles_vec2 = self.init_angles(self.n_nodes2)
        
        return self.integrate(angles_vec1, angles_vec2, adj_mat1, adj_mat2)
        # def derivative_complete(self, init, t, adj_mat1, adj_mat2, coupling1, coupling2, duration):
    #     start = time.time()
    #     res = derivative_numba_com_complete(init, t, adj_mat1, adj_mat2, coupling1, coupling2, self.natfreqs1, self.natfreqs2,
    #                                self.omega_f1, self.omega_f2, self.A, self.B, self.delta
    #     )
    #     end = time.time()
    #     duration.append(end - start)
    #     # print(f"elapsed time ={end-start}")
    #     return res
    
# @jit(nopython=True)
# def derivative_numba_com_complete(init, t, adj_mat1, adj_mat2, coupling1, coupling2, natfreqs1, natfreqs2, omega_f1, omega_f2, A, B, delta):    
#     '''
#     Modified version with the external field
#     Compute derivative of all nodes for current state, defined as

#         dx_i,1      natfreq_i,1 +  k1  sum_j,1 ( Aij,1 * sin (angle_j,1 - angle_i,1) ) + h * sin(omega_f1*t - angle_i,1)
#        -------   =                ----
#         dt                       M_i,1

#         dx_i,2      natfreq_i,2 +  k2  sum_j,2 ( Aij,2 * sin (angle_j,2 - angle_i,2) ) + h * sin(omega_f2*t - angle_i,2)
#        -------   =                 ----
#         dt                        M_i,2

#         dh      f(x_i,1, x_i,2)
#        ---- =             
#         dt                

#     '''
#     n_nodes1 = len(adj_mat1)
#     n_nodes2 = len(adj_mat2)
    
#     angles_vec1 = init[:n_nodes1]
#     angles_vec2 = init[n_nodes1:n_nodes1+n_nodes2]
#     ext = init[-5]
#     r1 = init[-4]
#     r2 = init[-3]
#     psi1 = init[-2]
#     psi2 = init[-1]
#     assert len(angles_vec1) == len(natfreqs1) == len(adj_mat1), \
#         'Input dimensions for the first matrix do not match, check lengths'
        
#     assert len(angles_vec2) == len(natfreqs2) == len(adj_mat2), \
#         'Input dimensions for the second matrix do not match, check lengths'
        
#     if np.abs(ext) >= 20:
#         a = 0.0 * np.ones(n_nodes1)
#         b = 0.0 * np.ones(n_nodes2)
#         c = 0.0 * np.ones(1)
#         d = np.concatenate((a, b, c, c, c, c, c))
#         return d
#     else:

#         # r1 = phase_coherence_numba(angles_vec1)
#         # r2 = phase_coherence_numba(angles_vec2)
#         int_new1 = coupling1 * n_nodes1 * np.imag(r1 * np.exp(-1j*angles_vec1))
#         int_new2 = coupling2 * n_nodes2 * np.imag(r2 * np.exp(-1j*angles_vec2))
        
#         interactions_ext1 = ext * np.sin(omega_f1 * t - angles_vec1)
#         interactions_ext2 = ext * np.sin(omega_f2 * t - angles_vec2)
#         dx1dt = natfreqs1 + int_new1 + interactions_ext1  # sum over incoming interactions (over columns)
#         dx2dt = natfreqs2 + int_new2 + interactions_ext2  # sum over incoming interactions (over columns)
        
#         dbdt1 = r1
#         dbdt2 = r2
#         dbdt3 = 0.0
        
#         for i in angles_vec1:
#             for j in angles_vec2:
#                 dbdt3 += (np.exp(1j*(i-j)))

#         dbdt3 = np.abs(dbdt3/(n_nodes1 * n_nodes2))

#         dbdt = -A*(dbdt1 + dbdt2)/2 + B*dbdt3
#         dbdt = np.array([(dbdt)], dtype = 'float32')

#         # dr1dt = -0.5*(r1**2*np.conjugate(coupling1 * r1 + ext) - (coupling1 * r1 + ext)) - (1 + 1j * omega_f1) * r1
#         # dr2dt = -0.5*(r2**2*np.conjugate(coupling2 * r2 + ext) - (coupling2 * r2 + ext)) - (1 + 1j * omega_f2) * r2
#         dr1dt = 0.5 * coupling1 * r1 * (1 - np.power(r1, 2)) - delta * r1 + 0.5 * ext * (1 - np.power(r1, 2)) * np.cos(psi1)
#         dr2dt = 0.5 * coupling2 * r2 * (1 - np.power(r2, 2)) - delta * r2 + 0.5 * ext * (1 - np.power(r2, 2)) * np.cos(psi2)
#         dpsi1dt = -(omega_f1 + 0.5 * ext * (r1 + 1/r1) * np.sin(psi1))
#         dpsi2dt = -(omega_f2 + 0.5 * ext * (r2 + 1/r2) * np.sin(psi2))
#         dr1dt = np.array([(dr1dt)], dtype = 'float32')
#         dr2dt = np.array([(dr2dt)], dtype = 'float32')
#         dpsi1dt = np.array([(dpsi1dt)], dtype = 'float32')
#         dpsi2dt = np.array([(dpsi2dt)], dtype = 'float32')
        
#         return np.concatenate((dx1dt, dx2dt, dbdt, dr1dt, dr2dt, dpsi1dt, dpsi2dt))
        # def derivative2(self, init, t, adj_mat1, adj_mat2, coupling1, coupling2):
    #     '''
    #     Modified version with the external field
    #     Compute derivative of all nodes for current state, defined as

    #     dx_i,1      natfreq_i,1 + k1  sum_j,1 ( Aij,1 * sin (angle_j,1 - angle_i,1) ) + h * sin(omega_f1*t - angle_i,1)
    #     ----   =                  ---
    #      dt                      M_i,1

    #     dx_i,2      natfreq_i,2 + k2  sum_j,2 ( Aij,2 * sin (angle_j,2 - angle_i,2) ) + h * sin(omega_f2*t - angle_i,2)
    #     ----   =                  ---
    #      dt                      M_i,2
 
    #      dh      f(x_i,1, x_i,2)
    #     ---- =             
    #      dt                

    #     '''
    #     angles_vec1 = init[:self.n_nodes1]
    #     angles_vec2 = init[self.n_nodes1:self.n_nodes1+self.n_nodes2]
    #     ext = init[-10]
    #     assert len(angles_vec1) == len(self.natfreqs1) == len(adj_mat1), \
    #         'Input dimensions for the first matrix do not match, check lengths'
            
    #     assert len(angles_vec2) == len(self.natfreqs2) == len(adj_mat2), \
    #         'Input dimensions for the second matrix do not match, check lengths'
            
    #     angles_i1, angles_j1 = np.meshgrid(angles_vec1, angles_vec1)
    #     interactions1 = adj_mat1 * np.sin(angles_j1 - angles_i1)  # Aij * sin(j-i)

    #     angles_i2, angles_j2 = np.meshgrid(angles_vec2, angles_vec2)
    #     interactions2 = adj_mat2 * np.sin(angles_j2 - angles_i2)  # Aij * sin(j-i)

    #     interactions_ext1 = ext * np.sin(self.omega_f1 * t - angles_vec1)
    #     interactions_ext2 = ext * np.sin(self.omega_f2 * t - angles_vec2)
        
    #     dx1dt = self.natfreqs1 + coupling1 * interactions1.sum(axis=0) + interactions_ext1  # sum over incoming interactions (over columns)
    #     dx2dt = self.natfreqs2 + coupling2 * interactions2.sum(axis=0) + interactions_ext2  # sum over incoming interactions (over columns)
    #     dbdt = [np.sin(np.sum(angles_vec1) + np.sum(angles_vec2))]
    #     return np.concatenate((dx1dt, dx2dt, dbdt))