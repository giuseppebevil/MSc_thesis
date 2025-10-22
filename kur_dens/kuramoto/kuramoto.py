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

class Kuramoto:

    def __init__(self, coupling=1, dt=0.01, t_max=10, n_nodes=None, natfreqs=None, coupling_ext = 0, omega_f = 1):
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
        

        if natfreqs is not None:
            np.random.seed(898289)
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
            np.random.seed(12345)
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