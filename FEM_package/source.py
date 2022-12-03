import numpy as np
import matplotlib.pyplot as plt

class source:
    def __init__(self, degree_fourier):
        self.deg = degree_fourier
    
    def sourceTrigRand(self, x, bound=100, NEW_COEFF=False):
        """Source function of sines

        Args:
            x (np.array): grid on which to solve 
            bound (int, optional): bound for coefficients, Defaults to 100.
            NEW_COEFF (bool, optional): Set to True if you want to create
            new coefficients for existing class. Defaults to False.

        Returns:
            Source function
        """        
        # Make sure coeff arr does not get overwritten
        if not hasattr(self, 'coeff_arr') or NEW_COEFF:
            self.coeff_arr = np.random.uniform(-bound, bound, self.deg)
            NEW_COEFF = False
   
        sin_arr = np.zeros((self.deg, len(x)))
        for i in range(self.deg):
            sin_arr[i] = np.sin((i+1)*np.pi*x)
        y = sin_arr * self.coeff_arr[:, np.newaxis]
        s = np.sum(y, axis=0)
        return s

    def sourceStep(self, x, NEW_COEFF=False): #Fourier series, step function
        """Creates step function

        Args:
            x (np.array): grid on which to solve 
            NEW_COEFF (bool, optional): Set to True if you want to create
            new coefficients for existing class. Defaults to False.

        Returns:
            Source function
        """        
        neu = self.deg * 8
        # Make sure coeff arr does not get overwritten
        if not hasattr(self, 'coeff_arr') or NEW_COEFF:
            self.coeff_arr = np.array([16*np.pi*(2*i+1) for i in range(self.deg)])
            NEW_COEFF = False
    
        sin_arr = np.zeros((self.deg, len(x)))
        for i in range(self.deg):
            sin_arr[i] = np.sin((2*i+1)*np.pi*x*2)*(2*i+1)
        y = sin_arr * 16 * np.pi
        s = np.sum(y, axis=0)
        return s

    def coeff_arr_out(self):
        """Outputs coefficients

        Returns:
            np.array: Coeffients array 
        """        
        try:
            return self.coeff_arr
        except:
            print("There exist no coeff arr")


