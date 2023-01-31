from scipy import integrate
import keras
import numpy as np
from utils import * 
from fem_solver import solver

class Recovery:
    def __init__(self, sol, mesh):
        self.hs = mesh[1:] - mesh[:-1]
        self.coeffs = np.zeros((len(self.hs) - 2, 3))
        x_midpoint = (mesh[:-1] + mesh[1:]) / 2
        self.grad_fem = (sol[1:] - sol[:-1]) / (mesh[1:]-mesh[:-1])
        for i in range(len(self.coeffs)):
             self.coeffs[i] = np.polyfit(x_midpoint[i:i+3], self.grad_fem[i:i+3], 2)
        
        first_row = self.coeffs[0]
        last_row = self.coeffs[-1]
        self.coeffs = np.insert(self.coeffs, 0, first_row, 0)
        self.coeffs = np.append(self.coeffs, [last_row], 0)
    
    def func(self, x):
        return self.coeffs[:, 0] * x**2 + self.coeffs[:, 1] * x + self.coeffs[:, 2]


def error_recovery(xi, sol, mesh, Recovery_err):
    u_ex_transform = Recovery_err.func(xi * (mesh[1:] - mesh[:-1]) + mesh[:-1])
    u_transform = (sol[1:] - sol[:-1]) / (mesh[1:] - mesh[:-1])
    return (u_ex_transform - u_transform)**2 * (mesh[1:] - mesh[:-1]) 


def recovery_error_estimator(sol, mesh):
    Recovery_err = Recovery(sol, mesh)
    energy_squared = integrate.quad_vec(error_recovery, 0, 1, args=(sol, mesh, Recovery_err))[0]
    energy_norm = np.sqrt(energy_squared)
    global_error = np.linalg.norm(energy_norm)
    return energy_norm, global_error


def explicit_error(a_k, freq, solution, mesh):
    step = mesh[1:] - mesh[:-1]
    r = element_residual(step, a_k, freq, mesh)[1:-1]
    j = get_jump(solution, step)
    step = step[1:-1]
    local_error = np.sqrt(step * step * r + 0.5 * step * j * j)
    local_error = np.insert(local_error, 0, local_error[0])
    local_error = np.append(local_error, local_error[-1])
    global_error = np.linalg.norm(local_error)
    return local_error, global_error


def build_explicit_error_estimator(bc, source_func):
    a_k = source_func[1]
    freq = source_func[2]
    estimator = partial(explicit_error, a_k, freq)
    return lambda solution, mesh: estimator(solution, mesh)


def richardson_error_estimator(bc, source_func, solution, mesh):
    # Solve on twice as fine 




