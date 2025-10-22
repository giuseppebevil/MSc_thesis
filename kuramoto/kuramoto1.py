import numpy as np
from scipy.integrate import odeint, solve_ivp
import random
import powerlaw
from datetime import datetime
random.seed(datetime.now().timestamp())
from numba import jit
from jitcode import jitcode, y
import time
from scipy.stats import cauchy
@jit(nopython=True)
def r_link_numba(adj_mat, act_mat, t_eq, delta_t, dt):
    # local order parameter:
    Nl = np.sum(adj_mat) #number of connections
    s = 0
    for i in range(len(adj_mat)):
        s1 = 0
        for j in range(len(adj_mat)):
            s2 = 0
            if adj_mat[i,j] != 0:
                for t in range(int(t_eq/dt), int((t_eq + delta_t)/dt)):
                    s2 += np.exp(1j* (act_mat[i,t] - act_mat[j,t])) * dt
            s1 += np.abs(s2/delta_t)
        s += s1
            
    # r = 1/Nl * s
    r = s/Nl
    
    return r

@jit(nopython=True)
def derivative_numba2(angles_vec, natfreqs, adj_mat, coupling, coupling_ext, omega_f, t):
    assert len(angles_vec) == len(natfreqs) == len(adj_mat), \
        'Input dimensions do not match, check lengths'
        
    #angles_i, angles_j = np.meshgrid(angles_vec, angles_vec)
    
    angles_i = np.reshape((np.repeat(angles_vec, len(angles_vec))), (len(angles_vec), len(angles_vec))).T
    angles_j = np.reshape((np.repeat(angles_vec, len(angles_vec))), (len(angles_vec), len(angles_vec)))

    #angles_i = np.stack([angles_vec] * len(angles_vec), axis = 0)
    #angles_j = np.stack([angles_vec] * len(angles_vec), axis = 1)
    
    #angles_i = np.vstack([angles_vec.reshape(1, len(angles_vec))] * len(angles_vec))
    #angles_j = np.hstack([angles_vec.reshape(1, len(angles_vec)).T] * len(angles_vec))

    interactions = adj_mat * np.sin(angles_j - angles_i)  # Aij * sin(j-i)    
    interactions_ext = coupling_ext * np.sin(omega_f * t - angles_vec)
    
    dxdt = natfreqs + coupling * interactions.sum(axis=0) + interactions_ext  # sum over incoming interactions (over columns)
    
    return dxdt

@jit(nopython=True)
def derivative_numba(angles_vec, natfreqs, adj_mat, coupling, coupling_ext, omega_f, t):    
    """
    Compute derivative of all nodes for current state, defined as:

    dx_i    natfreq_i + k  sum_j ( Aij* sin (angle_j - angle_i) ) + h * sin(omega_f*t - angle_i)
    ---- =             ---
     dt                M_i

    Parameters:
        angles_vec (np.ndarray): Current angles.
        natfreqs (np.ndarray): Natural frequencies.
        adj_mat (np.ndarray): Adjacency matrix of the graph.
        coupling (float): Coupling constant.
        coupling_ext (float): External coupling constant.
        omega_f (float): External field frequency.
        t (float): Current time.

    Returns:
        np.ndarray: Derivative values for the angles.
    """
    n_nodes = len(angles_vec)
    interactions = np.zeros(n_nodes)  # Initialize interaction array

    for i in range(n_nodes):
        for j in range(n_nodes):
            interactions[i] += adj_mat[i, j] * \
                np.sin(angles_vec[j] - angles_vec[i])

    interactions_ext = coupling_ext * np.sin(omega_f * t - angles_vec)
    dxdt = natfreqs + coupling * interactions + interactions_ext
    
    return dxdt

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

class Kuramoto:

    def __init__(self, coupling=1, dt=0.01, t_max=10, n_nodes=None, natfreqs=None, coupling_ext = 0, omega_f = 1, sigma = 1):
        '''
        coupling: float
            Coupling strength. Default = 1. Typical values range between 0.4-2
        dt: float
            Delta t for integration of equations.
        t_max: float
            Total time of simulated activity.
            From that the number of integration steps is t_max/dt.
        n_nodes: int, optional
            Number of oscillators.
            If None, it is inferred from len of natfreqs.
            Must be specified if natfreqs is not given.
        natfreqs: 1D ndarray, optional
            Natural oscillation frequencies.
            If None, then new random values will be generated and kept fixed
            for the object instance.
            Must be specified if n_nodes is not given.
            If given, it overrides the n_nodes argument.
        '''
        if n_nodes is None and natfreqs is None:
            raise ValueError("n_nodes or natfreqs must be specified")

        self.dt = dt
        self.t_max = t_max
        self.coupling = coupling
        self.coupling_ext = coupling_ext
        self.omega_f = omega_f
        self.sigma = sigma

        if natfreqs is not None:
            self.natfreqs = natfreqs
            self.n_nodes = len(natfreqs)
        else:
            """
            if self.coupling != 0:  # this part is to get a desired precision for the mean of the distribution
                x = False
                while x == False:
                    self.n_nodes = n_nodes
                    self.natfreqs = np.random.normal(size=self.n_nodes, loc = 0)
                    if(np.isclose(np.mean(self.natfreqs), 0.1)):
                        x = True           
            else:
            """ 
            self.n_nodes = n_nodes
            #self.natfreqs = np.random.normal(size = self.n_nodes, loc = 0, scale = sigma)
            self.natfreqs = np.random.uniform(size = self.n_nodes, low = -0.5, high = 0.5).astype('float32')
            #self.natfreqs = np.ones(self.n_nodes).astype('float32')

        #print("omega_0: ", np.mean(self.natfreqs))

    def init_angles(self):
        '''Random initial random angles (position, "theta").'''

        return 2 * np.pi * np.random.random(size=self.n_nodes).astype('float32')   
        #return np.zeros(self.n_nodes).astype('float32')    
    
    def derivative(self, angles_vec, t, adj_mat, coupling):
        #start = time.time()
        res = derivative_numba(
            angles_vec,
            self.natfreqs,
            adj_mat,
            coupling,
            self.coupling_ext,
            self.omega_f,
            t
        )
        #end = time.time()
        #print(f"elapsed time ={end-start}")
        return res
    
    def derivative1(self, angles_vec, t, adj_mat, coupling):
        #start = time.time()
        res = derivative_numba2(
            angles_vec,
            self.natfreqs,
            adj_mat,
            coupling,
            self.coupling_ext,
            self.omega_f,
            t
        )
        #end = time.time()
        #print(f"elapsed time ={end-start}")
        return res    
    
    def derivative2(self, angles_vec, t, adj_mat, coupling):
        '''
        Modified version with the external field
        Compute derivative of all nodes for current state, defined as

        dx_i    natfreq_i + k  sum_j ( Aij* sin (angle_j - angle_i) ) + h * sin(psi_i - angle_i)
        ---- =             ---
         dt                M_i

        dx_i    natfreq_i + k  sum_j ( Aij* sin (angle_j - angle_i) ) + h * sin(omega_f*t - angle_i)
        ---- =             ---
         dt                M_i

        t: for compatibility with scipy.odeint
        '''
        assert len(angles_vec) == len(self.natfreqs) == len(adj_mat), \
            'Input dimensions do not match, check lengths'

        #angles_i, angles_j = np.meshgrid(angles_vec, angles_vec)
        angles_i = np.reshape((np.repeat(angles_vec, len(angles_vec))), (len(angles_vec), len(angles_vec))).T
        angles_j = np.reshape((np.repeat(angles_vec, len(angles_vec))), (len(angles_vec), len(angles_vec)))
        interactions = adj_mat * np.sin(angles_j - angles_i)  # Aij * sin(j-i)
        
        #_, psi_i = np.meshgrid(angles_vec, angle_ext)
        #interactions_ext1 = amplitude * np.sin(psi_i - angles_i) # external field contribution
        
        interactions_ext = self.coupling_ext * np.sin(self.omega_f * t - angles_vec)
        dxdt = self.natfreqs + coupling * interactions.sum(axis=0) + interactions_ext  # sum over incoming interactions (over columns)
        return dxdt
    
    def integrate(self, angles_vec, adj_mat):
        '''Updates all states by integrating state of all nodes'''
        # Coupling term (k / Mj) is constant in the integrated time window.
        # Compute it only once here and pass it to the derivative function
        n_interactions = (adj_mat != 0).sum(axis=0)  # number of incoming interactions
        #coupling = self.coupling / max(n_interactions)  # normalize coupling by number of interactions, shape of coupling is (n_nodes,)
        coupling = self.coupling
        #fit = powerlaw.Fit(n_interactions) # power law fit of the degree sequence
        #gamma = fit.power_law.alpha # coefficient of the degree distribution
        #print('Gamma', gamma)
        #print('Coupling', self.coupling)
        #print('Maximum degree', max(n_interactions))
        #print('Number of interactions', np.sum(adj_mat))
        #print('Effective coupling', coupling)
        
        t = np.linspace(0, self.t_max, int(self.t_max/self.dt))
        timeseries = odeint(self.derivative, angles_vec, t, args=(adj_mat, coupling)) # shape of angles_vec is (n_nodes,)

        return timeseries.T  # transpose for consistency (act_mat:node vs time)
    
    def integrate1(self, angles_vec, adj_mat):
        coupling = self.coupling       
        t = np.linspace(0, self.t_max, int(self.t_max/self.dt))
        timeseries = odeint(self.derivative2, angles_vec, t, args=(adj_mat, coupling)) # shape of angles_vec is (n_nodes,)

        return timeseries.T  # transpose for consistency (act_mat:node vs time)

    def run(self, adj_mat=None, angles_vec=None):
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
        if angles_vec is None:
            angles_vec = self.init_angles()

        return self.integrate(angles_vec, adj_mat)

    def run1(self, adj_mat=None, angles_vec=None):
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
        if angles_vec is None:
            angles_vec = self.init_angles()

        return self.integrate1(angles_vec, adj_mat)

    def phase_coherence(angles_vec):
        '''
        Compute global order parameter R_t - mean length of resultant vector
        '''
        r = phase_coherence_numba(angles_vec)
        
        return r
       
    def r_link(adj_mat, act_mat, delta_t, t_eq, dt):
        r = r_link_numba(adj_mat, act_mat, t_eq, delta_t, dt)  
        return r
    
    def mean_frequency(self, act_mat, adj_mat):
        '''
        Compute average frequency within the time window (self.t_max) for all nodes
        '''
        assert len(adj_mat) == act_mat.shape[0], 'adj_mat does not match act_mat'
        _, n_steps = act_mat.shape

        # Compute derivative for all nodes for all time steps
        dxdt = np.zeros_like(act_mat)
        for time in range(n_steps):
            dxdt[:, time] = self.derivative(act_mat[:, time], None, adj_mat)

        # Integrate all nodes over the time window t_max
        integral = np.sum(dxdt * self.dt, axis=1)
        # Average across complete time window - mean angular velocity (freq.)
        meanfreq = integral / self.t_max
        return meanfreq
    
    def external_field(self):
        amplitude = self.coupling_ext
        angle1 = np.pi/3 * np.ones(int(self.n_nodes/2)) # if you don't put int it gives error
        angle2 = -np.pi/6 * np.ones(int(self.n_nodes/2))
        angle = np.concatenate((angle1, angle2))
        angle = np.linspace(0, 2*np.pi, self.n_nodes)
        
        return amplitude, angle

@jit(nopython=True)
def derivative_numba_com_complete(init, t, adj_mat1, adj_mat2, coupling1, coupling2, natfreqs1, natfreqs2, omega_f1, omega_f2, A, B, delta):    
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
    ext = init[-5]
    r1 = init[-4]
    r2 = init[-3]
    psi1 = init[-2]
    psi2 = init[-1]
    assert len(angles_vec1) == len(natfreqs1) == len(adj_mat1), \
        'Input dimensions for the first matrix do not match, check lengths'
        
    assert len(angles_vec2) == len(natfreqs2) == len(adj_mat2), \
        'Input dimensions for the second matrix do not match, check lengths'
        
    if np.abs(ext) >= 20:
        a = 0.0 * np.ones(n_nodes1)
        b = 0.0 * np.ones(n_nodes2)
        c = 0.0 * np.ones(1)
        d = np.concatenate((a, b, c, c, c, c, c))
        return d
    else:

        # r1 = phase_coherence_numba(angles_vec1)
        # r2 = phase_coherence_numba(angles_vec2)
        int_new1 = coupling1 * n_nodes1 * np.imag(r1 * np.exp(-1j*angles_vec1))
        int_new2 = coupling2 * n_nodes2 * np.imag(r2 * np.exp(-1j*angles_vec2))
        
        interactions_ext1 = ext * np.sin(omega_f1 * t - angles_vec1)
        interactions_ext2 = ext * np.sin(omega_f2 * t - angles_vec2)
        dx1dt = natfreqs1 + int_new1 + interactions_ext1  # sum over incoming interactions (over columns)
        dx2dt = natfreqs2 + int_new2 + interactions_ext2  # sum over incoming interactions (over columns)
        
        dbdt1 = r1
        dbdt2 = r2
        dbdt3 = 0.0
        
        for i in angles_vec1:
            for j in angles_vec2:
                dbdt3 += (np.exp(1j*(i-j)))

        dbdt3 = np.abs(dbdt3/(n_nodes1 * n_nodes2))

        dbdt = -A*(dbdt1 + dbdt2)/2 + B*dbdt3
        dbdt = np.array([(dbdt)], dtype = 'float32')

        dr1dt = 0.5 * coupling1 * r1 * (1 - np.power(r1, 2)) - delta * r1 + 0.5 * ext * (1 - np.power(r1, 2)) * np.cos(psi1)
        dr2dt = 0.5 * coupling2 * r2 * (1 - np.power(r2, 2)) - delta * r2 + 0.5 * ext * (1 - np.power(r2, 2)) * np.cos(psi2)
        dpsi1dt = -(omega_f1 + 0.5 * ext * (r1 + 1/r1) * np.sin(psi1))
        dpsi2dt = -(omega_f2 + 0.5 * ext * (r2 + 1/r2) * np.sin(psi2))
        dr1dt = np.array([(dr1dt)], dtype = 'float32')
        dr2dt = np.array([(dr2dt)], dtype = 'float32')
        dpsi1dt = np.array([(dpsi1dt)], dtype = 'float32')
        dpsi2dt = np.array([(dpsi2dt)], dtype = 'float32')
        
        return np.concatenate((dx1dt, dx2dt, dbdt, dr1dt, dr2dt, dpsi1dt, dpsi2dt))

@jit(nopython=True)
def derivative_numba_com(init, t, adj_mat1, adj_mat2, coupling1, coupling2, natfreqs1, natfreqs2, omega_f1, omega_f2, A, B, delta):    
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
    ext = init[-5]
    r1 = init[-4]
    r2 = init[-3]
    psi1 = init[-2]
    psi2 = init[-1]
    assert len(angles_vec1) == len(natfreqs1) == len(adj_mat1), \
        'Input dimensions for the first matrix do not match, check lengths'
        
    assert len(angles_vec2) == len(natfreqs2) == len(adj_mat2), \
        'Input dimensions for the second matrix do not match, check lengths'
        
    if np.abs(ext) >= 20:
        a = 0.0 * np.ones(n_nodes1)
        b = 0.0 * np.ones(n_nodes2)
        c = 0.0 * np.ones(1)
        d = np.concatenate((a, b, c, c, c, c, c))
        return d
    else:
        interactions1 = np.zeros(n_nodes1)  # Initialize interaction array
        interactions2 = np.zeros(n_nodes2)
        
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

        dbdt = -A*(dbdt1 + dbdt2)/2 + B*dbdt3
        dbdt = np.array([(dbdt)], dtype = 'float32')

        dr1dt = (0.5 * 12 * coupling1 * r1 * (1 - np.power(r1, 2)) - delta * r1 + 0.5 * ext * (1 - np.power(r1, 2)) * np.cos(psi1))*8
        dr2dt = (0.5 * 12 * coupling2 * r2 * (1 - np.power(r2, 2)) - delta * r2 + 0.5 * ext * (1 - np.power(r2, 2)) * np.cos(psi2))*8
        dpsi1dt = -(omega_f1 + 0.5 * ext * (r1 + 1/r1) * np.sin(psi1))
        dpsi2dt = -(omega_f2 + 0.5 * ext * (r2 + 1/r2) * np.sin(psi2))
        dr1dt = np.array([(dr1dt)], dtype = 'float32')
        dr2dt = np.array([(dr2dt)], dtype = 'float32')
        dpsi1dt = np.array([(dpsi1dt)], dtype = 'float32')
        dpsi2dt = np.array([(dpsi2dt)], dtype = 'float32')

        return np.concatenate((dx1dt, dx2dt, dbdt, dr1dt, dr2dt, dpsi1dt, dpsi2dt))

class Kuramoto_com:

    def __init__(self, coupling1=1.0, coupling2 = 1.0, dt=0.01, t_max=10, n_nodes1=None, natfreqs1=None,
                 n_nodes2=None, natfreqs2=None, omega_f1 = 1.0, omega_f2 = 1.0, A = 1, B = 5, delta = 1):
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

        
        if natfreqs1 is not None:
            self.natfreqs1 = natfreqs1
            self.n_nodes1 = len(natfreqs1)
        else:
            self.n_nodes1 = n_nodes1
            #self.natfreqs1 = cauchy.rvs(size = n_nodes1, loc = 0, scale = self.delta).astype('float32')
            self.natfreqs1 = np.random.uniform(size = self.n_nodes1, low = -0.5, high = 0.5).astype('float32')
            #self.natfreqs1 = np.random.normal(size = self.n_nodes1, loc = 0)

        if natfreqs2 is not None:
            self.natfreqs2 = natfreqs2
            self.n_nodes2 = len(natfreqs2)
        else:
            self.n_nodes2 = n_nodes2
            #self.natfreqs2 = cauchy.rvs(size = n_nodes2, loc = 0, scale = self.delta).astype('float32')
            self.natfreqs2 = np.random.uniform(size = self.n_nodes2, low = -0.5, high = 0.5).astype('float32')
            #self.natfreqs2 = np.random.normal(size = self.n_nodes2, loc = 0)

    def init_angles(self, n_nodes):
        '''
        Random initial random angles (position, "theta").
        '''
        return 2 * np.pi * np.random.random(size = n_nodes).astype('float32')


    def derivative2(self, init, t, adj_mat1, adj_mat2, coupling1, coupling2):
        '''
        Modified version with the external field
        Compute derivative of all nodes for current state, defined as

        dx_i,1      natfreq_i,1 + k1  sum_j,1 ( Aij,1 * sin (angle_j,1 - angle_i,1) ) + h * sin(omega_f1*t - angle_i,1)
        ----   =                  ---
         dt                      M_i,1

        dx_i,2      natfreq_i,2 + k2  sum_j,2 ( Aij,2 * sin (angle_j,2 - angle_i,2) ) + h * sin(omega_f2*t - angle_i,2)
        ----   =                  ---
         dt                      M_i,2
 
         dh      f(x_i,1, x_i,2)
        ---- =             
         dt                

        '''
        angles_vec1 = init[:self.n_nodes1]
        angles_vec2 = init[self.n_nodes1:self.n_nodes1+self.n_nodes2]
        ext = init[-10]
        assert len(angles_vec1) == len(self.natfreqs1) == len(adj_mat1), \
            'Input dimensions for the first matrix do not match, check lengths'
            
        assert len(angles_vec2) == len(self.natfreqs2) == len(adj_mat2), \
            'Input dimensions for the second matrix do not match, check lengths'
            
        angles_i1, angles_j1 = np.meshgrid(angles_vec1, angles_vec1)
        interactions1 = adj_mat1 * np.sin(angles_j1 - angles_i1)  # Aij * sin(j-i)

        angles_i2, angles_j2 = np.meshgrid(angles_vec2, angles_vec2)
        interactions2 = adj_mat2 * np.sin(angles_j2 - angles_i2)  # Aij * sin(j-i)

        interactions_ext1 = ext * np.sin(self.omega_f1 * t - angles_vec1)
        interactions_ext2 = ext * np.sin(self.omega_f2 * t - angles_vec2)
        
        dx1dt = self.natfreqs1 + coupling1 * interactions1.sum(axis=0) + interactions_ext1  # sum over incoming interactions (over columns)
        dx2dt = self.natfreqs2 + coupling2 * interactions2.sum(axis=0) + interactions_ext2  # sum over incoming interactions (over columns)
        dbdt = [np.sin(np.sum(angles_vec1) + np.sum(angles_vec2))]
        return np.concatenate((dx1dt, dx2dt, dbdt))
    
    def derivative(self, init, t, adj_mat1, adj_mat2, coupling1, coupling2):
        # start = time.time()
        res = derivative_numba_com(init, t, adj_mat1, adj_mat2, coupling1, coupling2, self.natfreqs1, self.natfreqs2,
                                   self.omega_f1, self.omega_f2, self.A, self.B, self.delta
        )
        # end = time.time()
        # print(f"elapsed time ={end-start}")
        
        return res
    
    def derivative_complete(self, init, t, adj_mat1, adj_mat2, coupling1, coupling2, duration):
        start = time.time()
        res = derivative_numba_com_complete(init, t, adj_mat1, adj_mat2, coupling1, coupling2, self.natfreqs1, self.natfreqs2,
                                   self.omega_f1, self.omega_f2, self.A, self.B, self.delta
        )
        end = time.time()
        duration.append(end - start)
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
        
        psi1_0 = 0
        psi2_0 = 0
        r1 = np.array([(r1)], dtype = 'float32')
        r2 = np.array([(r2)], dtype = 'float32')
        psi1_0 = np.array([(psi1_0)], dtype = 'float32')
        psi2_0 = np.array([(psi2_0)], dtype = 'float32')
        
        t = np.linspace(0, self.t_max, int(self.t_max/self.dt))
        init = np.concatenate((angles_vec1, angles_vec2, b_0, r1, r2, psi1_0, psi2_0))
        
        duration = []
        check = True #if both graphs are complete i use a faster expression to integrate
        for i in range(self.n_nodes1):
            for j in range(self.n_nodes1):
                if j != i and adj_mat2[i,j]!=1:
                    check = False
        for i in range(self.n_nodes2):
            for j in range(self.n_nodes2):
                if j != i and adj_mat2[i,j]!=1:
                    check = False
        if check == True:
            print('Complete graphs')
            timeseries = odeint(self.derivative_complete, init, t, args=(adj_mat1, adj_mat2, coupling1, coupling2, duration))
        else:
            print('Not complete graphs')
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
            angles_vec1 = self.init_angles(self.n_nodes1)
        if angles_vec2 is None:
            angles_vec2 = self.init_angles(self.n_nodes2)
            
        return self.integrate(angles_vec1, angles_vec2, adj_mat1, adj_mat2)

    @staticmethod
    def phase_coherence(angles_vec):
        '''
        Compute global order parameter R_t - mean length of resultant vector
        '''
        suma = sum([(np.e ** (1j * i)) for i in angles_vec])
        ord = np.sqrt(np.sum(np.cos(angles_vec[i]))**2 + np.sum(np.sin(angles_vec[i]))**2  for i in angles_vec)
        return abs(suma / len(angles_vec)), abs(ord / len(angles_vec))

    def mean_frequency(self, act_mat, adj_mat):
        '''
        Compute average frequency within the time window (self.t_max) for all nodes
        '''
        assert len(adj_mat) == act_mat.shape[0], 'adj_mat does not match act_mat'
        _, n_steps = act_mat.shape

        # Compute derivative for all nodes for all time steps
        dxdt = np.zeros_like(act_mat)
        for time in range(n_steps):
            dxdt[:, time] = self.derivative(act_mat[:, time], None, adj_mat)

        # Integrate all nodes over the time window t_max
        integral = np.sum(dxdt * self.dt, axis=1)
        # Average across complete time window - mean angular velocity (freq.)
        meanfreq = integral / self.t_max
        return meanfreq