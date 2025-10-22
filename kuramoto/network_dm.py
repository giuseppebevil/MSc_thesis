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
def H_J(coupling1, coupling2, angles_vec1, angles_vec2, b, adj_mat1, adj_mat2, omega_f, A, B, t):
    N1 = len(adj_mat1)
    N2 = len(adj_mat2)
    J = np.zeros((N1+N2+1, N1+N2+1))
    for i in range(N1):
        for j in range(N1):
            if i == j:
                for k in range(N1):
                    if k!=i:
                        J[i,j] -= coupling1*adj_mat1[i,k]*np.cos(angles_vec1[k] - angles_vec1[i])
                J[i,j] -= b * np.cos(omega_f * t - angles_vec1[i])
            else:
                J[i,j] = coupling1*adj_mat1[i,j]* np.cos(angles_vec1[j] - angles_vec1[i])
    for i in range(N2):
        for j in range(N2):
            if i == j:
                for k in range(N2):
                    if k!=i:
                        J[N1+i,N1+j] -= coupling2*adj_mat2[i,k]*np.cos(angles_vec2[k] - angles_vec2[i])
                J[N1+i,N1+j] -= b * np.cos(omega_f * t - angles_vec2[i])
            else:
                J[N1+i,N1+j] = coupling2*adj_mat2[i,j]* np.cos(angles_vec2[j] - angles_vec2[i])
    
    # Elements relative to b
    for i in range(N1):
        # numerator of the first term
        num1_1 = 0
        d1_1 = 0
        d2_1 = 0
        for j in range(N1):
            if j !=i:
                num1_1 += np.cos(angles_vec1[i])*np.sin(angles_vec1[j])
                num1_1 -= np.sin(angles_vec1[i])*np.cos(angles_vec1[j])
            d1_1 += np.cos(angles_vec1[j])
            d2_1 += np.sin(angles_vec1[j])
        # denominator of the first term
        den1_1 = np.sqrt(d1_1**2 + d2_1**2)
        #den1_1 = np.sqrt(np.sum(np.cos(angles_vec1[j]) for j in range(N1))**2 + np.sum(np.sin(angles_vec1[j]) for j in range(N1))**2)
        s1_1 = A/(2*N1)*num1_1/den1_1
        
        # denominator of the second term
        a1 = 0
        b1 = 0
        for h in range(N1):
            for k in range(N1):
                a1 += np.sin(angles_vec1[h] - angles_vec2[k])
                b1 += np.cos(angles_vec1[h] - angles_vec2[k])
        den2_1 = np.sqrt(a1**2 + b1**2)
        # numerator of the second term
        num2_1 = 0
        for j in range(N1):
            num2_1 += np.cos(angles_vec1[i]-angles_vec2[j])*a1
            num2_1 -= np.sin(angles_vec1[i]-angles_vec2[j])*b1
        s2_1 = B/(N1*N2) * num2_1/den2_1
        J[N1+N2, i] = s1_1 + s2_1
        J[i, N1+N2] = np.sin(omega_f*t - angles_vec1[i])
    for i in range(N2):
        num1_2 = 0
        d1_2 = 0
        d2_2 = 0
        for j in range(N2):
            if j !=i:
                num1_2 += np.cos(angles_vec2[i])*np.sin(angles_vec2[j])
                num1_2 -= np.sin(angles_vec2[i])*np.cos(angles_vec2[j])
            d1_2 += np.cos(angles_vec2[j])
            d2_2 += np.sin(angles_vec2[j])
        den1_2 = np.sqrt(d1_2**2 + d2_2**2)
        #den1_2 = np.sqrt(np.sum(np.cos(angles_vec2[j]) for j in range(N2))**2 + np.sum(np.sin(angles_vec2[j]) for j in range(N2))**2)
        s1_2 = A/2*(1/N2 * num1_2/den1_2)

        # denominator of the second term
        a2 = 0
        b2 = 0
        for h in range(N2):
            for k in range(N2):
                a2 += np.sin(angles_vec1[h] - angles_vec2[k])
                b2 += np.cos(angles_vec1[h] - angles_vec2[k])
        den2_2 = np.sqrt(a2**2 + b2**2)
        # numerator of the second term
        num2_2 = 0
        for j in range(N1):
            num2_2 -= np.cos(angles_vec1[j]-angles_vec2[i])*a2
            num2_2 += np.sin(angles_vec1[j]-angles_vec2[i])*b2
        s2_2 = B/(N1*N2) * num2_2/den2_2        
        
        J[N1+N2, N1+i] = s1_2 + s2_2
        J[N1+i, N1+N2] = np.sin(omega_f*t - angles_vec2[i])

    return -J

@jit(nopython=True)
def H_J_single1(coupling1, coupling2, angles_vec1, angles_vec2, b, adj_mat1, adj_mat2, omega_f, A, B, t):
    N1 = len(adj_mat1)
    N2 = len(adj_mat2)
    J = np.zeros((N1+1, N1+1))
    for i in range(N1):
        for j in range(N1):
            if i == j:
                for k in range(N1):
                    if k!=i:
                        J[i,j] -= coupling1*adj_mat1[i,k]*np.cos(angles_vec1[k] - angles_vec1[i])
                J[i,j] -= b * np.cos(omega_f * t - angles_vec1[i])
            else:
                J[i,j] = coupling1*adj_mat1[i,j]* np.cos(angles_vec1[j] - angles_vec1[i])
    
    # Elements relative to b
    for i in range(N1):
        # numerator of the first term
        num1_1 = 0
        d1_1 = 0
        d2_1 = 0
        for j in range(N1):
            if j !=i:
                num1_1 += np.cos(angles_vec1[i])*np.sin(angles_vec1[j])
                num1_1 -= np.sin(angles_vec1[i])*np.cos(angles_vec1[j])
            d1_1 += np.cos(angles_vec1[j])
            d2_1 += np.sin(angles_vec1[j])
        # denominator of the first term
        den1_1 = np.sqrt(d1_1**2 + d2_1**2)
        #den1_1 = np.sqrt(np.sum(np.cos(angles_vec1[j]) for j in range(N1))**2 + np.sum(np.sin(angles_vec1[j]) for j in range(N1))**2)
        s1_1 = A/(2*N1)*num1_1/den1_1
        
        # denominator of the second term
        a1 = 0
        b1 = 0
        for h in range(N1):
            for k in range(N1):
                a1 += np.sin(angles_vec1[h] - angles_vec2[k])
                b1 += np.cos(angles_vec1[h] - angles_vec2[k])
        den2_1 = np.sqrt(a1**2 + b1**2)
        # numerator of the second term
        num2_1 = 0
        for j in range(N1):
            num2_1 += np.cos(angles_vec1[i]-angles_vec2[j])*a1
            num2_1 -= np.sin(angles_vec1[i]-angles_vec2[j])*b1
        s2_1 = B/(N1*N2) * num2_1/den2_1
        J[N1, i] = s1_1 + s2_1
        J[i, N1] = np.sin(omega_f*t - angles_vec1[i])

    return -J

@jit(nopython=True)
def H_J_single2(coupling1, coupling2, angles_vec1, angles_vec2, b, adj_mat1, adj_mat2, omega_f, A, B, t):
    N1 = len(adj_mat1)
    N2 = len(adj_mat2)
    J = np.zeros((N2+1, N2+1))

    for i in range(N2):
        for j in range(N2):
            if i == j:
                for k in range(N2):
                    if k!=i:
                        J[i,j] -= coupling2*adj_mat2[i,k]*np.cos(angles_vec2[k] - angles_vec2[i])
                J[i,j] -= b * np.cos(omega_f * t - angles_vec2[i])
            else:
                J[i,j] = coupling2*adj_mat2[i,j]* np.cos(angles_vec2[j] - angles_vec2[i])
    
    # Elements relative to b
 
    for i in range(N2):
        num1_2 = 0
        d1_2 = 0
        d2_2 = 0
        for j in range(N2):
            if j !=i:
                num1_2 += np.cos(angles_vec2[i])*np.sin(angles_vec2[j])
                num1_2 -= np.sin(angles_vec2[i])*np.cos(angles_vec2[j])
            d1_2 += np.cos(angles_vec2[j])
            d2_2 += np.sin(angles_vec2[j])
        den1_2 = np.sqrt(d1_2**2 + d2_2**2)
        #den1_2 = np.sqrt(np.sum(np.cos(angles_vec2[j]) for j in range(N2))**2 + np.sum(np.sin(angles_vec2[j]) for j in range(N2))**2)
        s1_2 = A/2*(1/N2 * num1_2/den1_2)

        # denominator of the second term
        a2 = 0
        b2 = 0
        for h in range(N2):
            for k in range(N2):
                a2 += np.sin(angles_vec1[h] - angles_vec2[k])
                b2 += np.cos(angles_vec1[h] - angles_vec2[k])
        den2_2 = np.sqrt(a2**2 + b2**2)
        # numerator of the second term
        num2_2 = 0
        for j in range(N1):
            num2_2 -= np.cos(angles_vec1[j]-angles_vec2[i])*a2
            num2_2 += np.sin(angles_vec1[j]-angles_vec2[i])*b2
        s2_2 = B/(N1*N2) * num2_2/den2_2        
        
        J[N2, i] = s1_2 + s2_2
        J[i, N2] = np.sin(omega_f*t - angles_vec2[i])

    return -J

@jit(nopython=True)
def H_J_simple(coupling1, coupling2, angles_vec1, angles_vec2, adj_mat1, adj_mat2, b, omega_f, t):
    N1 = len(adj_mat1)
    N2 = len(adj_mat2)
    J = np.zeros((N1+N2, N1+N2))
    for i in range(N1):
        for j in range(N1):
            if i == j:
                for k in range(N1):
                    if k!=i:
                        J[i,j] -= coupling1*adj_mat1[i,k]*np.cos(angles_vec1[k] - angles_vec1[i])
                J[i,j] -= b * np.cos(omega_f * t - angles_vec1[i])
            else:
                J[i,j] = coupling1*adj_mat1[i,j]* np.cos(angles_vec1[j] - angles_vec1[i])
    for i in range(N2):
        for j in range(N2):
            if i == j:
                for k in range(N2):
                    if k!=i:
                        J[N1+i,N1+j] -= coupling2*adj_mat2[i,k]*np.cos(angles_vec2[k] - angles_vec2[i])
                J[i,j] -= b * np.cos(omega_f * t - angles_vec2[i])
            else:
                J[N1+i,N1+j] = coupling2*adj_mat2[i,j]* np.cos(angles_vec2[j] - angles_vec2[i])
    
    return -J

@jit(nopython=True)
def H_J_single_simple(coupling, angles_vec,adj_mat, b, omega_f, t):
    N = len(adj_mat)
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                for k in range(N):
                    if k!=i:
                        J[i,j] -= coupling*adj_mat[i,k]*np.cos(angles_vec[k] - angles_vec[i])
                J[i,j] -= b * np.cos(omega_f * t - angles_vec[i])
            else:
                J[i,j] = coupling*adj_mat[i,j]* np.cos(angles_vec[j] - angles_vec[i])
    
    return -J

def network_thermodynamics_S(H, tau):
    N = len(H)
    id = np.identity(N)
    HHT = H + H.T
    #U_t = statistical_propagator(H, tau)
    U_t = sci.linalg.expm(-tau * HHT)
    Z = np.trace(U_t)  # partition function
    rho = U_t/Z  # density matrix
    S = np.trace(rho @ ((tau * HHT) + id*np.log(Z)))/np.log(N)
    
    # log_rho = sci.linalg.logm(rho).real
    # S = -np.trace(np.matmul(rho, log_rho)).real/np.log(N) # entropy
    # F = -np.log(Z)/tau
    # W = F + np.log(N)/tau
    # Q = (S - np.log(N))/tau
    # dF = np.log(N) - np.log(Z)
    # dS = np.sum( -eigvals * np.log(eigvals) ) - np.log(N)
    # eta = (W + Q)/W

    return S

def network_thermodynamics_C(H, tau):
    N = len(H)
    id = np.identity(N)
    HHT = H + H.T
    #U_t = statistical_propagator(H, tau)
    U_t = sci.linalg.expm(-tau * HHT)
    Z = np.trace(U_t)  # partition function
    tr2 = np.trace(HHT @ sci.linalg.expm(-tau * HHT))
    C = tau * np.trace(U_t/Z @ (-HHT + id * tr2/Z) + (U_t * tr2/Z**2 - HHT@U_t/Z)@(-tau*HHT- id*np.log(Z)))/np.log(N)

    return C

def statistical_propagator(H, tau):
    continuous = True
    old = False
    ### Returns {density matrix, entropy, partition function, free energy}
    if continuous:
        G_tau = sci.linalg.expm(-tau * H)
    #else:
        #G_tau = G_tau_discrete(H,tau)
    ### With the assumption that all nodes have equal probability of perturbation
    if old:
        U = G_tau / len(H)
    else:
        U = np.matmul(G_tau, G_tau.T)
    return U