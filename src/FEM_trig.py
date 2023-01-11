# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 20:25:48 2022

@author: Alvaro Baillet Boliv
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.sparse as sp
import scipy.sparse.linalg as la
from numpy.random import default_rng
import random
import time

def sourceTrigRand(x, coeff_arr, degree_fourier):
    sin_arr = np.zeros((degree_fourier, len(x)))
    for i in range(degree_fourier):
        sin_arr[i] = np.sin((i+1)*np.pi*x)
    y = sin_arr * coeff_arr[:, np.newaxis]
    s = np.sum(y, axis=0)
    return s

def sourceStep(x, coeff_arr, degree_fourier): #Fourier series, step function
    sin_arr = np.zeros((degree_fourier, len(x)))
    for i in range(degree_fourier):
        sin_arr[i] = np.sin((2*i+1)*np.pi*x*2)*(2*i+1)
    y = sin_arr * 16 * np.pi
    s = np.sum(y, axis=0)
    return s

def residual(xi, hs, xgrid, sourceFunc, coeff_arr, degree_fourier):
    return sourceFunc(hs*xi + xgrid[:-1], coeff_arr, degree_fourier)**2 * hs

def exactSol(x, coeff_arr, neu, degree_fourier):
    integrated_coefs1 = 1/(np.pi*np.arange(1, degree_fourier+1, 1))*-coeff_arr
    a = np.zeros(degree_fourier)
    a[::2] = -1
    a[1::2] = 1
    c = neu + (np.sum(integrated_coefs1 * a))
    integrated_coefs2 = 1/(np.pi**2*np.arange(1, degree_fourier+1, 1)**2)*-coeff_arr
    sin_arr = np.zeros((degree_fourier, len(x)))
    for i in range(degree_fourier):
        sin_arr[i] = np.sin((i+1)*np.pi*x)
    y = sin_arr * integrated_coefs2[:, np.newaxis]
    s = np.sum(y, axis=0) - c*x
    return -s

def gradTrigError(x, coeff_arr, neu, degree_fourier):
    integrated_coefs1 = 1/(np.pi*np.arange(1, degree_fourier+1, 1))*-coeff_arr
    a = np.zeros(degree_fourier)
    a[::2] = -1
    a[1::2] = 1
    c = neu + (np.sum(integrated_coefs1 * a))
    cos_arr = np.zeros((degree_fourier, len(x)))
    for i in range(degree_fourier):
        cos_arr[i] = np.cos((i+1)*np.pi*x)
    y = cos_arr * integrated_coefs1[:, np.newaxis]
    s = np.sum(y, axis=0) - c
    return -s

def gradStep(x, coeff_arr, neu, degree_fourier):
    cos_arr = np.zeros((degree_fourier, len(x)))
    for i in range(degree_fourier):
        cos_arr[i] = np.cos((2*i+1)*np.pi*x*2)
    y = cos_arr * 8
    s = np.sum(y, axis=0)
    return -s

def hatLMaster(xi, hs, xgrid, sourceFunc, coeff_arr, degree_fourier):
    '''
    

    Parameters
    ----------
    xi : A variable obtained by the coordinate transformation: xi = (x-x0)/(x1-x0), 
    x0 is the left coordiate of the element, x1 the right point of the element.

    Returns
    -------
    the quantity w*f evaluated on the master element.
    
    This function is used to compute the integral of wf, where w is the weight function
    and f is the source function. It does so on a master element xi in [0, 1].
    The coordinate transformation is xi = (x-x0)/(x1-x0).
    '''
    return xi*sourceFunc(hs*xi + xgrid[:-1], coeff_arr, degree_fourier)*hs

def hatRMaster(xi, hs, xgrid, sourceFunc, coeff_arr, degree_fourier):
    # same as hat l but correpsonds to the right part of the hat basis function.
    return (1-xi)*sourceFunc(hs[1:]*xi + xgrid[1:-1], coeff_arr, degree_fourier)*hs[1:]

    
def assembleRHSVec(xgrid, neu, sourceFunc, hs, coeff_arr, degree_fourier):
    '''
    

    Parameters
    ----------
    xgrid : grid.
    neu : neumann condition.

    Returns
    -------
    rhs : f vector, of fem system Ax = f.

    '''
    I1 = integrate.quad_vec(hatLMaster, 0, 1, args=(hs, xgrid, sourceFunc, coeff_arr, degree_fourier))[0]
    I2 = integrate.quad_vec(hatRMaster, 0, 1, args=(hs, xgrid, sourceFunc, coeff_arr, degree_fourier))[0]
    I2 = np.append(I2, neu)
    rhs = I1 + I2
    return rhs

def assembleFEMVec(hs):
    '''
    

    Parameters
    ----------
    hs : array of grid spacing, used to assemble the FEM matrix.

    Returns
    -------
    A : Matrix discretisation of poisson operator.

    '''
    main_diag = 1/hs[:-1] + 1/hs[1:]
    main_diag = np.append(main_diag, 1/hs[-1])
    off_diag = -1/hs[1:]
    
    A = sp.diags([main_diag, off_diag, off_diag], [0, 1, -1], format="csc")

    return A

def errorMaster(xi, hs, xgrid, grad_fem, coeff_arr, neu, degree_fourier):
    return (gradTrigError(xi*hs + xgrid[:-1], coeff_arr, neu, degree_fourier)-grad_fem)**2 * hs


def plot(u, x, t, markeropt=None):
    plt.plot(x, u, marker=markeropt, label=t)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u$")
    plt.legend()
    plt.grid(True)
    plt.show()
    

def genData(N_elements, sourceFunc, neu, sampling_freq, degree_fourier=10):
    N_interior = N_elements + 1    
    xgrid = np.linspace(0, 1, N_interior)
    hs = xgrid[1:]-xgrid[:-1]

    
    if sourceFunc == sourceStep:
        neu = degree_fourier * 8
        coeff_arr = np.array([16*np.pi*(2*i+1) for i in range(degree_fourier)])
    elif sourceFunc == sourceTrigRand:
        coeff_arr = np.random.rand(degree_fourier)
        multipliers = np.random.randint(-400, 400, degree_fourier)
        coeff_arr *= multipliers
    #print(neu)

    A = assembleFEMVec(hs)
    rhs = assembleRHSVec(xgrid, neu, sourceFunc, hs, coeff_arr, degree_fourier)
    u_fem = la.spsolve(A, rhs) #Solve system
    u_fem = np.insert(u_fem, 0, 0) #Impose homogenous dir condition on lhs u(0)=0
    
    grad_fem = (u_fem[1:] - u_fem[:-1])/(xgrid[1:] - xgrid[:-1]) #Compute gradient of each element.
    jump_left = grad_fem[1:] - grad_fem[:-1]
    jump_right = jump_left
    jump_left = np.insert(jump_left, 0, 0)
    jump_right = np.append(jump_right, neu-grad_fem[-1])
    jump_sq = jump_left*jump_left + jump_right*jump_right
    #print("Jumps sq: ", jump_sq[:10])
        
    error_norms_sq = integrate.quad_vec(errorMaster, 0, 1, args=(hs, xgrid, grad_fem, coeff_arr, neu, degree_fourier))[0] #compute (energy norm)^2 per element
    local_errors = np.sqrt(error_norms_sq) #compute energy norm per element
    #print("Local Errors: ", local_errors) #Not correct for step (need exact step sol)
    
    residuals_sq = integrate.quad_vec(residual, 0, 1, args=(hs, xgrid, sourceFunc, coeff_arr, degree_fourier))[0]
    local_residuals = np.sqrt(residuals_sq)
    #print("Local Residual: ", local_residuals[:10])

    global_error = np.sqrt(np.sum(error_norms_sq)) #compute global error
    #print("Global error: {:.5e}".format(global_error)) #not correct for step

    # =============================================================================
    hs_im1 = hs[0:-2:sampling_freq]
    hs_i = hs[1:-1:sampling_freq]
    hs_ip1 = hs[2::sampling_freq]
    
    res_im1 = residuals_sq[0:-2:sampling_freq]
    res_i = residuals_sq[1:-1:sampling_freq]
    res_ip1 = residuals_sq[2::sampling_freq]
    
    jump_im1 = jump_sq[0:-2:sampling_freq]
    jump_i = jump_sq[1:-1:sampling_freq]
    jump_ip1 = jump_sq[2::sampling_freq]
    
    data = np.vstack((hs_im1, hs_i, hs_ip1, res_im1, res_i, res_ip1, jump_im1, jump_i, jump_ip1, error_norms_sq[1:-1:sampling_freq]))
    data = np.transpose(data)
    #print(data)
    
    with open("features.csv", "ab") as f:
        np.savetxt(f, data, delimiter=",")

    # =============================================================================
    
    
    #plot(u_fem, xgrid, "FEM solution")


#genData(100000, sourceTrigRand, random.uniform(-5, 5), 400, 10)

N_samples = 3000
base_funcs = 10

elements = [2**i for i in range(3, 18)]

N_funcs = []
freq_lst = []
for i in range(len(elements)):
    req = N_samples/(elements[i])
    if req <= base_funcs:
        req = 10
    N_funcs.append(int(req))
    sample_freq = int(elements[i]/(N_samples/base_funcs)) + 1
    freq_lst.append(sample_freq)
    
s = sum(N_funcs)
    
x = 0
for i in range(len(N_funcs)):
    N_elements = elements[i]
    frequency = freq_lst[i]
    for j in range(N_funcs[i]):
        genData(N_elements, sourceTrigRand, random.uniform(-5, 5), frequency, degree_fourier=10)
        x += 1
        print(x,"/",s)