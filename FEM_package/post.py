import numpy as np
from scipy import integrate

class post:
    def __init__(self, xgrid, sourceFunc, coeff_arr, degree_fourier, neu, u_fem):
        self.xgrid = xgrid
        self.sourceFunc = sourceFunc
        self.coeff_arr = coeff_arr
        self.deg = degree_fourier
        self.neu = neu
        self.hs = xgrid[1:] - xgrid[:-1]
        self.grad_fem = (u_fem[1:] - u_fem[:-1])/(xgrid[1:] - xgrid[:-1]) 
    
    def residual(self, xi):
        return self.sourceFunc(self.hs*xi + self.xgrid[:-1])**2 * self.hs

    def exactSol(self, *args):
        """Output exact solutions 

        Returns:
            tuple: np.array fine grid, np.array y values
        """        
        
        x = np.linspace(0, 1, 10000)
        if args == 1:
            x = args[0]
            
        integrated_coefs1 = 1/(np.pi*np.arange(1, self.deg+1, 1))*-self.coeff_arr
        a = np.zeros(self.deg)
        a[::2] = -1
        a[1::2] = 1
        c = self.neu + (np.sum(integrated_coefs1 * a))
        integrated_coefs2 = 1/(np.pi**2*np.arange(1, self.deg+1, 1)**2)*-self.coeff_arr
        sin_arr = np.zeros((self.deg, len(x)))
        for i in range(self.deg):
            sin_arr[i] = np.sin((i+1)*np.pi*x)
        y = sin_arr * integrated_coefs2[:, np.newaxis]
        s = np.sum(y, axis=0) - c*x
        return (x, -s)

    def gradTrigError(self, x):              
        integrated_coefs1 = 1/(np.pi*np.arange(1, self.deg+1, 1))*-self.coeff_arr
        a = np.zeros(self.deg)
        a[::2] = -1
        a[1::2] = 1
        c = self.neu + (np.sum(integrated_coefs1 * a))
        cos_arr = np.zeros((self.deg, len(x)))
        for i in range(self.deg):
            cos_arr[i] = np.cos((i+1)*np.pi*x)
        y = cos_arr * integrated_coefs1[:, np.newaxis]
        s = np.sum(y, axis=0) - c
        return -s
    
    def gradStep(self, x):
        cos_arr = np.zeros((self.deg, len(x)))
        for i in range(self.deg):
            cos_arr[i] = np.cos((2*i+1)*np.pi*x*2)
        y = cos_arr * 8
        s = np.sum(y, axis=0)
        return -s

    def errorMaster(self, xi):
        return (self.gradTrigError(xi*self.hs + self.xgrid[:-1])-self.grad_fem)**2 * self.hs

    def energy(self):
        self.energy = integrate.quad_vec(self.errorMaster, 0, 1)[0]
        self.energy_norm = np.sqrt(self.energy)
        return self.energy_norm
    
    def residual_err(self):
        self.residual_norm = integrate.quad_vec(self.residual, 0, 1)[0]
        return self.residual_norm
    
    def jump(self):
        jump_left = self.grad_fem[1:] - self.grad_fem[:-1]
        jump_right = jump_left
        jump_left = np.insert(jump_left, 0, 0)
        jump_right = np.append(jump_right, self.neu-self.grad_fem[-1])
        self.jump_sq = jump_left*jump_left + jump_right*jump_right
        return self.jump_sq

    def gendata(self, sampling_freq):
        """generate data

        Args:
            sampling_freq int: Determines values taken out solution

        Returns:
            np.array: data
        """        
        hs_im1 = self.hs[0:-2:sampling_freq]
        hs_i = self.hs[1:-1:sampling_freq]
        hs_ip1 = self.hs[2::sampling_freq]
        
        res_im1 = self.residual_norm[0:-2:sampling_freq]
        res_i = self.residual_norm[1:-1:sampling_freq]
        res_ip1 = self.residual_norm[2::sampling_freq]
        
        jump_im1 = self.jump_sq[0:-2:sampling_freq]
        jump_i = self.jump_sq[1:-1:sampling_freq]
        jump_ip1 = self.jump_sq[2::sampling_freq]
        
        data = np.vstack((hs_im1, hs_i, hs_ip1, res_im1, res_i, res_ip1, jump_im1, jump_i, jump_ip1, self.energy[1:-1:sampling_freq]))
        data = np.transpose(data)
        return data


