import keras
import numpy as np
from functools import partial

def get_error_estimate(features):
    neural_network = keras.models.load_model("models/12_features_scaled_100.ckpt")
    local_errors = neural_network.predict(features)
    global_error = np.linalg.norm(local_errors)
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
    hs = hs.reshape(-1, 1)
    hs = hs[1:-1]
    f_sample = sampleSource(func, xgrid, frac)
    g_sample = gradient(u_fem, xgrid)
    data = np.hstack((f_sample, g_sample, hs))
    return data



def adaptive_mesh_refinement():
    mesh = np.linspace(0, 1, 9)
    source_func_temp = f_str(100, 100, 100)
    source_func_str = source_func_temp[0]
    source_func = partial(f_exp, source_func_temp[1], source_func_temp[2])
    bc = np.random.uniform(-10, 10)
    tolerance = 1e-3
    error = 1 << 20
    while error > tolerance:
        solution = solver(mesh, bc, source_func_str)
        features = get_features(source_func, mesh, solution)
        local_error, error = get_error_estimate(features)
        mesh = refine(mesh, local_error, error)
    return solution
