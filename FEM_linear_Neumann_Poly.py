# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 00:21:16 2022

@author: Alvaro Baillet Boliv
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
from scipy import integrate
import random
import sys
import time

def docs():
    f = open("docs_FEM.txt")
    lines = f.readlines()
    f.close()
    print("#"*10, "\n")
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
        print(lines[i])
    print("\n")
    print("#"*10)
    
def sourceVisual(coefs_source): # This is a function to print the source function, nothing fancy
    print("f = ",np.polynomial.polynomial.Polynomial(coefs_source))
    
def sourcePoly(x): 
    '''
    

    Parameters
    ----------
    x : float or array.
    coefs_source: coefficients of the polynomial

    Returns
    -------
    source function evaluated at x.

    '''
    global coefs_source
    return np.polynomial.polynomial.Polynomial(coefs_source)(x)


def exactPoly(x, coefs, neu):
    '''
    

    Parameters
    ----------
    x : grid.
    coefs : coefficients of source function.
    neu : neumann boundary condition.

    Returns
    -------
    exact solution and exact gradient.
    This function is only used for plotting later, it uses the numpy Polynomial class 
    to compute exact solution and exact gradient given the coefficients of the source function.

    '''
    source = np.polynomial.polynomial.Polynomial(coefs)
    integ1 = -np.polynomial.polynomial.Polynomial.integ(source, m=1, k=[0])
    b = neu - integ1(1)
    exact = -np.polynomial.polynomial.Polynomial.integ(source, m=2, k=[-b, 0])
    exact_grad = np.polynomial.polynomial.Polynomial.deriv(exact, m=1)
    return exact(x), exact_grad(x)


def gradPolyError(x):
    '''
    

    Parameters
    ----------
    x : grid.

    Returns
    -------
    Exact gradient.
    This function is exactly the same as exactPoly, except it only return the 
    exact gradient. This function will be used for numerical integration later.
    '''
    global coefs_source, q
    source = np.polynomial.polynomial.Polynomial(coefs_source)
    integ1 = -np.polynomial.polynomial.Polynomial.integ(source, m=1, k=[0])
    b = q - integ1(1)
    exact = -np.polynomial.polynomial.Polynomial.integ(source, m=2, k=[-b, 0])
    exact_grad = np.polynomial.polynomial.Polynomial.deriv(exact, m=1)
    
    return exact_grad(x)

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


def hatLMaster(xi):
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
    global hs, xgrid
    return xi*sourcePoly(hs*xi + xgrid[:-1])*hs

def hatRMaster(xi):
    # same as hat l but correpsonds to the right part of the hat basis function.
    global hs, xgrid
    return (1-xi)*sourcePoly(hs[1:]*xi + xgrid[1:-1])*hs[1:]

    
def assembleRHSVec(xgrid, neu):
    '''
    

    Parameters
    ----------
    xgrid : grid.
    neu : neumann condition.

    Returns
    -------
    rhs : f vector, of fem system Ax = f.

    '''
    I1 = integrate.quad_vec(hatLMaster, 0, 1)[0]
    I2 = integrate.quad_vec(hatRMaster, 0, 1)[0]
    I2 = np.append(I2, neu)
    rhs = I1 + I2
    return rhs
    
def errorMaster(xi):
    global hs, xgrid, grad_fem
    return (gradPolyError(xi*hs + xgrid[:-1])-grad_fem)**2 * hs

    
def plot(u, x, t, markeropt=None):
    plt.plot(x, u, marker=markeropt, label=t)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u$")
    plt.legend()
    plt.show()
    
docs()

degree_source = random.randint(1, 10) #randomly generate degree of source polynomial.

coefs_source = np.random.rand(degree_source) #randomly generate coefficients of polynomial
multipliers = np.random.randint(0, 10, degree_source) #randomly multiply those coefficients by some number
coefs_source *= multipliers

sourceVisual(coefs_source) #print source function

N_interior = 120 #number of internal points

finegrid = np.arange(0, 1.000001, 0.001) #fine grid used for plotting

s = time.time()

rng = default_rng()
type_grid = "rand"  
#if reg, generates a regular grid with N interior points, 
#if rand, generates a random grid with N interior points
#if fine, generates a fine grid (if we want to use a very fine solution)
if type_grid == "reg":
    xgrid = np.linspace(0, 1, N_interior)
elif type_grid == "rand":
    xgrid = np.sort(rng.choice(finegrid[10:], size=N_interior, replace=False))
    xgrid = np.append(xgrid, 1)
    xgrid = np.insert(xgrid, 0, 0)
elif type_grid == "fine":
    xgrid = finegrid
else:
    sys.exit()

hs = xgrid[1:]-xgrid[:-1]  #computes grid spacing
#print(hs)
print("Grid: ",xgrid[:10])

# Neumann condition
q = 1
print("u_x(1) =", q)

A = assembleFEMVec(hs)  #Assemble A
rhs = assembleRHSVec(xgrid, q) #Assemble right hand side

print("condition number: {:.3e}".format(np.linalg.cond(A.toarray()))) #Compute condition number

u_fem = la.spsolve(A, rhs) #Solve system
u_fem = np.insert(u_fem, 0, 0) #Impose homogenous dir condition on rhs u(0)=0


grad_fem = (u_fem[1:] - u_fem[:-1])/(xgrid[1:] - xgrid[:-1]) #Compute gradient of each element.


error_norms_sq = integrate.quad_vec(errorMaster, 0, 1)[0] #compute (energy norm)^2 per element
local_errors = np.sqrt(error_norms_sq) #compute energy norm per element
print("Local Errors: ", local_errors[-10:-1]) #print last 10

global_error = np.sqrt(np.sum(error_norms_sq)) #compute global error
print("Global error: ",global_error)

u_exfine, u_gradfine = exactPoly(finegrid, coefs_source, q)


plot(u_fem, xgrid, "FEM solution", "o")
plot(u_exfine, finegrid, "Exact solution")
plot(u_gradfine, finegrid, "Exact derivative")
print(" --- {:.6f} seconds---".format(time.time()-s))