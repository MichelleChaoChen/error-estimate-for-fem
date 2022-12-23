from numpy import sqrt, arange, round

EPSILON = 1e-4
import keras
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import source
import solver as own_solver
import post
def get_error_estimate(features):
    nn_fine = keras.models.load_model("models/fine_training_scaled.ckpt")
    nn_coarse = keras.models.load_model("models/coarse_training_scaled.ckpt")
    threshold = 2**-12
    fine_separation = np.logical_and(features[:, 0] < threshold, features[:, 1] < threshold, features[:, 2] < threshold)

    try:
        print('using fine')
        fine_error = nn_fine.predict(features[fine_separation][:, 3:]).flatten()
    except:
        fine_error = 0
    
    try:
        print('using coarse')
        coarse_error = nn_coarse.predict(features[~fine_separation][:, 3:]).flatten()
    except:
        coarse_error = 0


    local_errors = np.zeros(len(features))
    local_errors[fine_separation] = fine_error
    local_errors[~fine_separation] = coarse_error 
    local_errors = local_errors * np.sqrt(features[:, 1]) / 10.0
    local_errors = np.insert(local_errors, 0, local_errors[0])
    local_errors = np.append(local_errors, local_errors[-1]) 

    global_error = np.linalg.norm(np.array(local_errors))
    return local_errors, global_error


##########################################
# Feature generation
##########################################
def sampleSource(func, xgrid, frac):
    hs = xgrid[1:] - xgrid[:-1]
    x_sample_source = np.tile(frac, (len(xgrid),1))
    hs_copy = np.append(hs, 1)
    x_copy = xgrid.copy()
    x_copy = np.reshape(x_copy, (len(xgrid), 1))
    
    x_sample_source = x_sample_source * hs_copy[:, np.newaxis]
    x_sample_source = x_sample_source + x_copy
    x_sample_source = x_sample_source.flatten()
    
    f_samples = func(x_sample_source)
    f_samples = np.reshape(f_samples, (len(xgrid), len(frac)))[:-1]


    f_im1 = f_samples[0:-2]
    f_i = f_samples[1:-1]
    f_ip1 = f_samples[2:]
    
    func_sample = np.hstack((f_im1, f_i, f_ip1))
    return func_sample


def gradient(u_fem, xgrid):
    grad_fem = (u_fem[1:] - u_fem[:-1])/(xgrid[1:] - xgrid[:-1]) 
    grad_im1 = grad_fem[0:-2]
    grad_i = grad_fem[1:-1]
    grad_ip1 = grad_fem[2:]
    
    grad_sample = np.vstack((grad_im1, grad_i, grad_ip1))
    grad_sample = np.transpose(grad_sample)
    return grad_sample


def get_features(func, xgrid, u_fem, frac=[0.25, 0.5, 0.75]):  
    hs = xgrid[1:] - xgrid[:-1]
    hs_i = hs[1:-1]
    hs_m1 = hs[0:-2]
    hs_p1 = hs[2:]
    #print(hs_i, hs_m1, hs_p1)
    
    hs_i = hs_i.reshape(-1, 1)
    hs_m1 = hs_m1.reshape(-1, 1)
    hs_p1 = hs_p1.reshape(-1, 1)
    
    f_sample = sampleSource(func, xgrid, frac)
    g_sample = gradient(u_fem, xgrid)
    data = np.hstack((hs_m1, hs_i, hs_p1, f_sample, g_sample))
    return data


# Function that creates random sourcefunction
def f_str(coeff_range, freq_range, N):
    a_k = np.random.uniform(-coeff_range, coeff_range, N)
    freq = np.pi * np.random.randint(1, freq_range, N)
    my_string = ''
    for i in range(N):
        string = "%s*sin(%s*x[0])" % (str(a_k[i]), str(freq[i]))
        if i != N - 1:
            string += ' + '
        my_string += string
    return [my_string, a_k, freq]


# Define source function non-string format
def f_exp(a_k, freq, x):
    f = 0
    for i in range(len(a_k)):
        f += a_k[i] * np.sin(freq[i] * x)
    return f

def exact_sol(x, a_k, freq, neu):
    C_1 = neu - np.sum((a_k / freq) * np.cos(freq))
    result = C_1 * x
    for i in range(len(a_k)):
        result += (a_k[i] / freq[i]**2) * np.sin(freq[i] * x)  
    return result

def solver(mesh_new, deg, f_source_func, neu):
    xgrid = mesh_new
    # Generate source
   # Store coefficients of source

    # Solver
    solver_solve = own_solver.solver(xgrid, None, f_source_func)  # Setup solver
    solver_solve.assembleFEMVec()                                       # Assemble matrix A
    solver_solve.assembleRHSVec(neu)                                    # Assemble RHS
    u_sol = solver_solve.solve()  
        
    return u_sol

def refine(mesh, err_pred, global_error):
    #mesh = mesh[1:len(mesh) - 1]
    # base case
    if len(mesh) == 1:
        return mesh

    num_elements = len(mesh)
    compute_err = lambda err: err * sqrt(num_elements) / global_error

    refined_mesh = [mesh[0]]
    for i in range(0, num_elements - 1):
        curErr = compute_err(err_pred[i])
        num_points = int(round(curErr))

        refined_mesh.extend(
            mesh[i] + (mesh[i + 1] - mesh[i]) / (num_points + 1) * arange(1, num_points + 2)
        )

        if np.isclose(curErr, 0.5, atol=EPSILON) and \
                (i + 1 < len(err_pred) and np.isclose(compute_err(err_pred[i + 1]), 0.5, atol=EPSILON)):
            refined_mesh.pop()
    # refined_mesh = np.insert(refined_mesh, 0, 0)
    # refined_mesh = np.append(refined_mesh, 1)
    return np.array(refined_mesh)


def adaptive_mesh_refinement():
    mesh = np.linspace(0, 1, 9)
    source_func_temp = f_str(1000, 15, 3)
    source_func_str = source_func_temp[0]
    source_func = partial(f_exp, source_func_temp[1], source_func_temp[2])
    tolerance = 1e-3
    error = 1 << 20
    iter = 0
    neu = np.random.uniform(-500, 500)
    x = np.linspace(0, 1, 100000)
    u_exact = exact_sol(x, source_func_temp[1], source_func_temp[2], neu)
    while error > tolerance:
        iter += 1
        solution = solver(mesh, 10, source_func, neu)
        features = get_features(source_func, mesh, solution)
        local_error, error = get_error_estimate(features)
        print(" ERROR ", error)
        plt.figure()
        plt.title(f"Iteration: {iter}, Error: {error}")
        plt.plot(x, u_exact)
        plt.plot(mesh, solution, 'r.-')

        plt.savefig(f"OWN: Iteration {iter}")
        mesh = refine(mesh, local_error, error)
        
    return solution

if __name__ == '__main__':
    adaptive_mesh_refinement()


