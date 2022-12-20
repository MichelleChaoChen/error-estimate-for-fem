import keras
import numpy as np
from functools import partial

def get_error_estimate(features):
    neural_network = keras.models.load_model("models/12_features_scaled_100.ckpt")
    local_errors = neural_network.predict(features)
    global_error = np.linalg.norm(local_errors)
    return local_errors, global_error


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
