import numpy as np
from scipy.integrate import odeint, solve_ivp
import random
import powerlaw
from datetime import datetime
# random.seed(datetime.now().timestamp())
#random.seed(1)
from numba import jit
from jitcode import jitcode, y
import time
from scipy.stats import cauchy

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
def phase_coherence_mod_numba(angles_vec, deg):
    '''
    Compute global order parameter R_t - mean length of resultant vector
    '''
    suma = 0.0
    deg = np.array(deg, dtype='float64')
    for i in range(len(angles_vec)):
        suma = suma + (deg[i] * np.exp(1j*angles_vec[i]))
        
    r = np.abs(suma / (len(angles_vec) * np.mean(deg)))

    return r

@jit(nopython=True)
def ris(ri_1, ri_2, betas, N1, N2, deg_1, deg_2, n_steps):
    '''
    Compute global order parameter R_t - mean length of resultant vector
    '''

    r1 = []
    r2 = []
    for beta in betas:
        rb_1 = np.zeros(n_steps)
        rb_2 = np.zeros(n_steps)
        
        d_1 = 0.0
        d_2 = 0.0
        for i in range(N1):
            d_1 += deg_1[i]**beta
            d_2 += deg_2[i]**beta

        for i in range(n_steps):
            for j in range(N1):
                rb_1[i] += ri_1[i,j] * deg_1[j]**beta
        for i in range(n_steps):
            for j in range(N2):
                rb_2[i] += ri_2[i,j] * deg_2[j]**beta
        rb_1 = rb_1 / d_1
        rb_2 = rb_2 / d_1
        
        r1.append(rb_1)
        r2.append(rb_2)

    return r1, r2

@jit(nopython=True)
def global_order_param(act_mat, adj_mat, n_steps, beta):
    '''
    Compute global order parameter R_t - mean length of resultant vector
    '''
    N = len(adj_mat)

    ri = np.zeros((N, n_steps), dtype = 'complex64')
    deg = np.zeros(N)
    for i in range(N):
        for j in range(N):
            deg[i] += adj_mat[i,j]
    # deg = np.sum(adj_mat[:,i] for i in range(N))
    for i in range(n_steps):
        angles_vec = act_mat[:,i]
        for j in range(N):
            for k in range(N):
                if adj_mat[j,k] != 0:
                    ri[j,i] += np.exp(1j*(angles_vec[k]))
    for i in range(N):
        if deg[i] != 0:
            ri[i] = np.abs(ri[i])/deg[i]
    ri = np.abs(ri)
    r = np.zeros(n_steps)
    d = 0.0
    for i in range(N):
        d += deg[i]**beta
    #d = np.sum(deg[i]**beta for i in range (N))
    for i in range(n_steps):
        for j in range(N):
            r[i] += ri[j,i] * deg[j]**beta

    r = r / d

    return r

@jit(nopython=True)
def derivative_numba_com(init, t, adj_mat1, adj_mat2, coupling1, coupling2, natfreqs1, natfreqs2, omega_f1, omega_f2, A, B, delta, d1, d2):    
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
    angles_vec2 = init[n_nodes1:n_nodes1 + n_nodes2]
    ri1 = init[n_nodes1 + n_nodes2 :2*n_nodes1 + n_nodes2]
    ri2 = init[2*n_nodes1 + n_nodes2 :2*n_nodes1 + 2*n_nodes2]
    psii1 = init[2*n_nodes1 + 2* n_nodes2 : 3*n_nodes1 + 2*n_nodes2]
    psii2 = init[3*n_nodes1 + 2*n_nodes2 :3*n_nodes1 + 3*n_nodes2]
    r1 = init[-5]
    r2 = init[-4]
    psi1 = init[-3]
    psi2 = init[-2]
    ext = init[-1]
    assert len(angles_vec1) == len(natfreqs1) == len(adj_mat1), \
        'Input dimensions for the first matrix do not match, check lengths'
        
    assert len(angles_vec2) == len(natfreqs2) == len(adj_mat2), \
        'Input dimensions for the second matrix do not match, check lengths'
        
    if np.abs(ext) >= 20:
        print('###############################')
        a = 0.0 * np.ones(n_nodes1)
        b = 0.0 * np.ones(n_nodes2)
        c = 0.0 * np.ones(1)
        d = np.concatenate((a, b, a, b, a, b, c, c, c, c, c))
        return d
    else:
        interactions1 = np.zeros(n_nodes1, dtype = 'float32')  # Initialize interaction array
        interactions2 = np.zeros(n_nodes2, dtype = 'float32')
        
        for i in range(n_nodes1):
            for j in range(n_nodes1):
                interactions1[i] += coupling1 * adj_mat1[i, j] * np.sin(angles_vec1[j] - angles_vec1[i])
        for i in range(n_nodes2):
            for j in range(n_nodes2):
                interactions2[i] += coupling2 * adj_mat2[i, j] * np.sin(angles_vec2[j] - angles_vec2[i])

        interactions_ext1 = ext * np.sin(omega_f1 * t - angles_vec1)
        interactions_ext2 = ext * np.sin(omega_f2 * t - angles_vec2)
        dx1dt = natfreqs1 + interactions1 + interactions_ext1  # sum over incoming interactions (over columns)
        dx2dt = natfreqs2 + interactions2 + interactions_ext2  # sum over incoming interactions (over columns)
        
        dri1dt = np.zeros(n_nodes1, dtype = 'float32')
        for i in range(n_nodes1):
            dri1dt[i] = 0.5 * coupling1 * d1[i] * ri1[i] * (1 - np.power(ri1[i], 2)) - delta * ri1[i] + 0.5 * ext * (1 - np.power(ri1[i], 2)) * np.cos(psii1[i])
            
        dr1dt = 0.5 * coupling1 * 6 * r1 * (1 - np.power(r1, 2)) - delta * r1 + 0.5 * ext  * (1 - np.power(r1, 2)) * np.cos(psi1)
        
        dri2dt = np.zeros(n_nodes2, dtype = 'float32')
        for i in range(n_nodes2):
            dri2dt[i] = 0.5 * coupling2 * d2[i] * ri2[i] * (1 - np.power(ri2[i], 2)) - delta * ri2[i] + 0.5 * ext * (1 - np.power(ri2[i], 2)) * np.cos(psii2[i])
        
        dr2dt = 0.5 * coupling2 * 6 * r2 * (1 - np.power(r2, 2)) - delta * r2 + 0.5 * ext  * (1 - np.power(r2, 2)) * np.cos(psi2)

        dpsii1dt = np.zeros(n_nodes1, dtype = 'float32')
        for i in range(n_nodes1):
            if ri1[i] !=0:
                dpsii1dt[i] = -(omega_f1 + 0.5 * ext * (ri1[i] + 1/ri1[i]) * np.sin(psii1[i]))
            else:
                dpsii1dt[i] = 0
                
        dpsi1dt = -(omega_f1 + 0.5 * ext * (r1 + 1/r1) * np.sin(psi1))
        
        dpsii2dt = np.zeros(n_nodes2, dtype = 'float32')
        for i in range(n_nodes2):
            if ri2[i] !=0:
                dpsii2dt[i] = -(omega_f2 + 0.5 * ext * (ri2[i] + 1/ri2[i]) * np.sin(psii2[i]))
            else:
                dpsii2dt[i] = 0
        
        dpsi2dt = -(omega_f2 + 0.5 * ext * (r2 + 1/r2) * np.sin(psi2))
        
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
        
        dr1dt = np.array([(dr1dt)], dtype = 'float32')
        dr2dt = np.array([(dr2dt)], dtype = 'float32')
        dpsi1dt = np.array([(dpsi1dt)], dtype = 'float32')
        dpsi2dt = np.array([(dpsi2dt)], dtype = 'float32')
        
        return np.concatenate((dx1dt, dx2dt, dri1dt, dri2dt, dpsii1dt, dpsii2dt, dr1dt, dr2dt, dpsi1dt, dpsi2dt, dbdt))

class Kuramoto_com:

    def __init__(self, coupling1=1.0, coupling2 = 1.0, dt=0.01, t_max=10, n_nodes1=None, natfreqs1=None,
                 n_nodes2=None, natfreqs2=None, omega_f1 = 1.0, omega_f2 = 1.0, A = 1, B = 5, delta = 1, deg1 = 0, deg2 = 0):
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
        self.delta = delta
        self.coupling1 = coupling1
        self.coupling2 = coupling2
        self.omega_f1 = omega_f1
        self.omega_f2 = omega_f2
        self.deg1 = deg1
        self.deg2 = deg2
        if natfreqs1 is not None:
            self.natfreqs1 = natfreqs1
            self.n_nodes1 = len(natfreqs1)
        else:
            np.random.seed(898289)
            self.n_nodes1 = n_nodes1
            self.natfreqs1 = cauchy.rvs(size = n_nodes1, loc = 0, scale = self.delta).astype('float32')
            #self.natfreqs1 = np.random.uniform(size = self.n_nodes1, low = -0.5, high = 0.5).astype('float32')
            #self.natfreqs1 = np.random.normal(size = self.n_nodes1, loc = 0)

        if natfreqs2 is not None:
            self.natfreqs2 = natfreqs2
            self.n_nodes2 = len(natfreqs2)
        else:
            np.random.seed(22322)
            self.n_nodes2 = n_nodes2
            self.natfreqs2 = cauchy.rvs(size = n_nodes2, loc = 0, scale = self.delta).astype('float32')
            #self.natfreqs2 = np.random.uniform(size = self.n_nodes2, low = -0.5, high = 0.5).astype('float32')
            #self.natfreqs2 = np.random.normal(size = self.n_nodes2, loc = 0)

    def init_angles(self, n_nodes):
        '''
        Random initial random angles (position, "theta").
        '''
        np.random.seed(585552)
        return 2 * np.pi * np.random.random(size = n_nodes).astype('float32')
    
    def derivative(self, init, t, adj_mat1, adj_mat2):
        # start = time.time()
        res = derivative_numba_com(init, t, adj_mat1, adj_mat2, self.coupling1, self.coupling2, self.natfreqs1, self.natfreqs2,
                                   self.omega_f1, self.omega_f2, self.A, self.B, self.delta, self.deg1, self.deg2
        )
        # end = time.time()
        # print(f"elapsed time ={end-start}")
        
        return res
        
    def integrate(self, angles_vec1, angles_vec2, adj_mat1, adj_mat2):
        '''Updates all states by integrating state of all nodes'''

        ri1 = np.zeros(self.n_nodes1, dtype = 'complex64')
        ri2 = np.zeros(self.n_nodes2, dtype = 'complex64')
        for i in range(self.n_nodes1):
            for j in range(self.n_nodes1):
                if adj_mat1[i,j] != 0:
                    ri1[i] += np.exp(1j*(angles_vec2[j]))
                    #ri1[i] = phase_coherence_numba(angles_vec1)
        for i in range(self.n_nodes1):
            for j in range(self.n_nodes2):
                if adj_mat2[i,j] != 0:
                    ri2[i] += np.exp(1j*(angles_vec2[j]))
                    #ri2[i] = phase_coherence_numba(angles_vec2)
        for i in range(self.n_nodes1):
            if self.deg1[i] != 0:
                ri1[i] = np.abs(ri1[i])/self.n_nodes1
                # ri1[i] = np.abs(ri1[i])/self.deg1[i]
        for i in range(self.n_nodes2): 
            if self.deg2[i] != 0:
                ri2[i] = np.abs(ri2[i])/self.n_nodes2
                # ri2[i] = np.abs(ri2[i])/self.deg2[i]
        ri1 = np.array(ri1, dtype = 'float32')
        ri2 = np.array(ri2, dtype = 'float32')

        psii1 = np.zeros(self.n_nodes1, dtype = 'float32')
        psii2 = np.zeros(self.n_nodes2, dtype = 'float32')

        r1 = phase_coherence_numba(angles_vec1)
        r2 = phase_coherence_numba(angles_vec2)
        psi1 = 0
        psi2 = 0
        r1 = np.array([(r1)], dtype = 'float32')
        r2 = np.array([(r2)], dtype = 'float32')
        psi1 = np.array([(psi1)], dtype = 'float32')
        psi2 = np.array([(psi2)], dtype = 'float32')
        b = 0
        b = np.array([(b)], dtype = 'float32') # initial condition for the coupling of the field
        
        t = np.linspace(0, self.t_max, int(self.t_max/self.dt))
        init = np.concatenate((angles_vec1, angles_vec2, ri1, ri2, psii1, psii2, r1, r2, psi1, psi2, b))
        timeseries = odeint(self.derivative, init, t, args=(adj_mat1, adj_mat2))
            
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
            angles_vec1 = self.init_angles(self.n_nodes1)
        if angles_vec2 is None:
            angles_vec2 = self.init_angles(self.n_nodes2)

        return self.integrate(angles_vec1, angles_vec2, adj_mat1, adj_mat2)