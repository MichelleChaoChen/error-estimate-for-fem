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
        self.ufem = u_fem
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
        u_ex_transform = self.gradTrigError(xi*self.hs + self.xgrid[:-1]) 
        u_fem_transform = self.grad_fem 
        return (u_ex_transform-u_fem_transform)**2 * self.hs

    def energy(self):
        self.energy_squared = integrate.quad_vec(self.errorMaster, 0, 1)[0]
        self.energy_norm = np.sqrt(self.energy_squared)
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
    
    def create_grad(self):
        self.coeffs = np.zeros((len(self.hs)-2, 3))
        x_midpoint = (self.xgrid[:-1] + self.xgrid[1:]) / 2
        
        for i in range(len(self.coeffs)):
            self.coeffs[i] = np.polyfit(x_midpoint[i:i+3], self.grad_fem[i:i+3], 2)
        
        first_row = self.coeffs[0]
        last_row = self.coeffs[-1]
        self.coeffs = np.insert(self.coeffs, 0, first_row, 0)
        self.coeffs = np.append(self.coeffs, [last_row], 0)
        print(self.coeffs.shape)

    def grad(self, x):
        return self.coeffs[:, 0] * x**2 + self.coeffs[:, 1] * x + self.coeffs[:, 2]

    def errorMasterRecovery(self, xi):
        u_rec_transform = self.grad(xi*self.hs + self.xgrid[:-1])  
        u_fem_transform = self.grad_fem 
        return (u_rec_transform - u_fem_transform)**2 * self.hs

    def energyRecovery(self):
        self.create_grad()
        self.energy_squared_recovery = integrate.quad_vec(self.errorMasterRecovery, 0, 1)[0]
        self.energy_norm_recovery = np.sqrt(self.energy_squared_recovery)
        return self.energy_norm_recovery



    ## BELOW ARE ALL GENDATA 
       
    def gendata1(self, sampling_freq, frac):
        """generate data 1

        Input: hs, (x, SF) * 9, grad, grad, grad, error
        Args:
            sampling_freq int: Determines values taken out solution

        Returns:
            np.array: data
        """        

        x_sample_source = np.tile(frac, (len(self.xgrid),1))
        hs_copy = np.append(self.hs, 1)
        x_copy = self.xgrid.copy()
        x_copy = np.reshape(x_copy, (len(self.xgrid), 1))
        
        x_sample_source = x_sample_source * hs_copy[:, np.newaxis]
        x_sample_source = x_sample_source + x_copy
        x_sample_source_copy = x_sample_source[:-1]
        
        x_sample_source = x_sample_source.flatten()
        
        f_samples = self.sourceFunc(x_sample_source)
        f_samples = np.reshape(f_samples, (len(self.xgrid), len(frac)))[:-1]
        
        
        hs_i = self.hs[1:-1:sampling_freq]

        x_im1 = x_sample_source_copy[0:-2:sampling_freq]
        f_im1 = f_samples[0:-2:sampling_freq]
        data_im1 = np.vstack((x_im1[:, 0], f_im1[:, 0], x_im1[:, 1], f_im1[:, 1], x_im1[:, 2], f_im1[:, 2]))
 
        x_i = x_sample_source_copy[1:-1:sampling_freq]
        f_i = f_samples[1:-1:sampling_freq]
        data_i = np.vstack((x_i[:, 0], f_i[:, 0], x_i[:, 1], f_i[:, 1], x_i[:, 2], f_i[:, 2]))

        x_ip1 = x_sample_source_copy[2::sampling_freq]
        f_ip1 = f_samples[2::sampling_freq]
        data_ip1 = np.vstack((x_ip1[:, 0], f_ip1[:, 0], x_ip1[:, 1], f_ip1[:, 1], x_ip1[:, 2], f_ip1[:, 2]))
        
        grad_im1 = self.grad_fem[0:-2:sampling_freq]
        grad_i = self.grad_fem[1:-1:sampling_freq]
        grad_ip1 = self.grad_fem[2::sampling_freq]
        #print(x_i.shape)
        #print(f_i.shape)
        data = np.vstack((hs_i, data_im1, data_i, data_ip1, grad_im1, grad_i, grad_ip1, self.energy_norm[1:-1:sampling_freq]))
        data = np.transpose(data)
        #data = np.hstack((f_im1, f_i, f_ip1, data))
        #print(data)

        #data = np.hstack((hs_i.reshape(len(hs_i), 1), data))
        return data


    def gendata2(self, sampling_freq):
        """generate data 2

        Input: hs, (x_midpoint, grad) * 3, u_fem, Error

        Args:
            sampling_freq int: Determines values taken out solution

        Returns:
            np.array: data
        """        

        node_1 = self.ufem[:-3:sampling_freq]
        node_2 = self.ufem[1:-2:sampling_freq]
        node_3 = self.ufem[2:-1:sampling_freq]
        node_4 = self.ufem[3::sampling_freq]

        x_midpoint = (self.xgrid[:-1] + self.xgrid[1:]) / 2
        x_im1 = x_midpoint[:-2:sampling_freq]
        x_i = x_midpoint[1:-1:sampling_freq]
        x_ip1 = x_midpoint[2::sampling_freq]

        hs_i = self.hs[1:-1:sampling_freq]
        grad_im1 = self.grad_fem[0:-2:sampling_freq]
        grad_i = self.grad_fem[1:-1:sampling_freq]
        grad_ip1 = self.grad_fem[2::sampling_freq]

        #print(x_i.shape)
        #print(f_i.shape)
        data = np.vstack((hs_i, x_im1, grad_im1, x_i, grad_i, x_ip1, grad_ip1, node_1, node_2, node_3, node_4, self.energy_norm[1:-1:sampling_freq]))
        data = np.transpose(data)
        #data = np.hstack((f_im1, f_i, f_ip1, data))
        #print(data)

        #data = np.hstack((hs_i.reshape(len(hs_i), 1), data))
        return data

    def gendata3(self, sampling_freq, frac):
        """generate data
        Args:
            sampling_freq int: Determines values taken out solution
        Returns:
            np.array: data
        """        

        x_sample_source = np.tile(frac, (len(self.xgrid),1))
        hs_copy = np.append(self.hs, 1)
        x_copy = self.xgrid.copy()
        x_copy = np.reshape(x_copy, (len(self.xgrid), 1))
        
        x_sample_source = x_sample_source * hs_copy[:, np.newaxis]
        x_sample_source = x_sample_source + x_copy
        x_sample_source = x_sample_source.flatten()
        
        f_samples = self.sourceFunc(x_sample_source)
        f_samples = np.reshape(f_samples, (len(self.xgrid), len(frac)))[:-1]
        hs_i = self.hs[1:-1:sampling_freq]

        f_im1 = f_samples[0:-2:sampling_freq]

        f_i = f_samples[1:-1:sampling_freq]
        f_ip1 = f_samples[2::sampling_freq]
        
        grad_im1 = self.grad_fem[0:-2:sampling_freq]
        grad_i = self.grad_fem[1:-1:sampling_freq]
        grad_ip1 = self.grad_fem[2::sampling_freq]
        
        data = np.vstack((hs_i, grad_im1, grad_i, grad_ip1, self.energy_norm[1:-1:sampling_freq]))
        data = np.transpose(data)
        data = np.hstack((f_im1, f_i, f_ip1, data))

        #data = np.hstack((hs_i.reshape(len(hs_i), 1), data))
        return data