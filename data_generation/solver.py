import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy import integrate
import scipy.sparse.linalg as la

class solver:
    def __init__(self, xgrid, coeff_arr, sourcefunc):
        self.xgrid = xgrid
        self.hs = xgrid[1:] - xgrid[:-1]
        self.sourcefunc = sourcefunc
        self.coeff_arr = coeff_arr
        
    def assembleFEMVec(self):
        '''
        

        Parameters
        ----------
        hs : array of grid spacing, used to assemble the FEM matrix.

        Returns
        -------
        A : Matrix discretisation of poisson operator.

        '''
        main_diag = 1/self.hs[:-1] + 1/self.hs[1:]
        main_diag = np.append(main_diag, 1/self.hs[-1])
        off_diag = -1/self.hs[1:]
        
        self.A = sp.diags([main_diag, off_diag, off_diag], [0, 1, -1], format="csc")

        return self.A
    
    def assembleRHSVec(self, neu):
        '''
        

        Parameters
        ----------
        xgrid : grid.
        neu : neumann condition.

        Returns
        -------
        rhs : f vector, of fem system Ax = f.

        '''
        
        I1 = integrate.quad_vec(self.hatLMaster, 0, 1)[0]
        I2 = integrate.quad_vec(self.hatRMaster, 0, 1)[0]
        I2 = np.append(I2, neu)
        self.rhs = I1 + I2

        return self.rhs
    
    def hatLMaster(self, xi):
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
        return xi*self.sourcefunc(self.hs*xi + self.xgrid[:-1])*self.hs

    def hatRMaster(self, xi):
        # same as hat l but correpsonds to the right part of the hat basis function.
        return (1-xi)*self.sourcefunc(self.hs[1:]*xi + self.xgrid[1:-1])*self.hs[1:]

    def solve(self):
        u_fem = la.spsolve(self.A, self.rhs)
        u_fem = np.insert(u_fem, 0, 0)
        return u_fem
