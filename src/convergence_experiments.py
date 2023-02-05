import keras

from adaptive_mesh_refinement import adaptive_mesh_refinement, build_nn_error_estimator
from plot_utilities import plot_mse_error_estimate, plot_number_elements, plot_average_iterations, plot_refinement
from utils import process_amr_data, write_amr_data, f_str, search_mesh_size, f_str_det, energy
from classical_methods import recovery_error_estimator, build_explicit_error_estimator
from fem_solver import solver 

import numpy as np 

def convergence_plots(nn_file, exp_file, rec_file):
    """
    Creates several plots that demonstrate the convergence of 
    adaptive mesh refinement using neural network as error estimator 
    and compares the performance with recovery-based estimator:
    1. Comparison of the MSE in error estimation of neural network and recovery method
    2. Comparison of the average mesh size over iterations
    3. Comparsion of the average number of iterations refinement needed

    :param nn_file: File containing AMR data with neural network
    :param exp_file: File containing AMR data with explicit error estimator
    :param rec_file: File containing AMR data with recovery-based error estimator
    """
    data_nn, avg_run_nn =  process_amr_data(nn_file)
    data_rec, avg_run_rec = process_amr_data(rec_file)
    data_exp, avg_run_exp = process_amr_data(exp_file)
    plot_mse_error_estimate(data_nn, data_exp, data_rec)
    plot_number_elements(data_nn, data_rec)
    plot_average_iterations(avg_run_nn, avg_run_rec)


def convergence_experiments(tolerance, max_iter, file_nn, file_explicit, file_recovery):
    """
    Verifies the convergence AMR with neural network and 
    compares performance with a recovery-based estimator. 

    :param tolerance: Global error tolerance
    :param max_iter: Maximum number of iterations of refinement
    """
    # Load models
    nn_fine = keras.models.load_model("models/Fine_NU_U_HF3GF3J2_logerr_base20_train.h5")
    nn_coarse = keras.models.load_model("models/Coarse_NU_U_HF3GF3J2_logerr_base20_train.h5")

    # Initialise AMR variables
    bc = 0
    source_func= f_str(1000, 40, 2)
    source_func_str = source_func[0]

    coarse_mesh, initial_mesh = search_mesh_size(40, bc, source_func_str)

    # AMR with neural network
    nn_error_estimator = build_nn_error_estimator(bc, source_func_str, nn_fine, nn_coarse, coarse_mesh)
    x_nn, solution_exact_nn, meshes_nn, solutions_nn, est_global_errors_nn, ex_global_errors_nn, N_elements_nn = \
         adaptive_mesh_refinement(tolerance, max_iter, bc, source_func, nn_error_estimator, initial_mesh)
    write_amr_data(file_nn, N_elements_nn, est_global_errors_nn, ex_global_errors_nn)

    # AMR with explicit error estimator
    explicit_error_estimator = build_explicit_error_estimator(bc, source_func)
    x_ex, solution_exact_ex, meshes_ex, solutions_ex, est_global_errors_ex, ex_global_errors_ex, N_elements_ex = \
        adaptive_mesh_refinement(tolerance, max_iter, bc, source_func, explicit_error_estimator, initial_mesh)
    write_amr_data(file_explicit, N_elements_ex, est_global_errors_ex, ex_global_errors_ex)

    # AMR with recovery error estimator
    x_rec, solution_exact_rec, meshes_rec, solutions_rec, est_global_errors_rec, ex_global_errors_rec, N_elements_rec = \
        adaptive_mesh_refinement(tolerance, max_iter, bc, source_func, recovery_error_estimator, initial_mesh)
    write_amr_data(file_recovery, N_elements_rec, est_global_errors_rec, ex_global_errors_rec)


def count_effectivity(freq, error_est, error_ext):
    ratio = error_est / error_ext
    cut_off = np.logical_or(ratio > 10, ratio < 0.1)
    frac = len(ratio[cut_off]) / len(ratio)
    return 100 * frac 


def test_frequency(max_iter):
    # Load models
    nn_fine = keras.models.load_model("models/Fine_NU_U_HF3GF3J2_logerr_base20_train.h5")
    nn_coarse = keras.models.load_model("models/Coarse_NU_U_HF3GF3J2_logerr_base20_train.h5")

    # Initialise AMR variables
    bc = 0
    low = 100
    high = 500
    freq = (low + high) / 2
    eff = 1 << 20 
    percent = 100
    i = 0 

    while abs(eff - percent) > 1 and i < max_iter:
        i += 1  
        source_func= f_str_det(freq)
        source_func_str = source_func[0]
        coarse_mesh, mesh = search_mesh_size(40, bc, source_func_str)
        error_estimator = build_nn_error_estimator(bc, source_func_str, nn_fine, nn_coarse, coarse_mesh)

        solution = solver(mesh, bc, source_func_str)
        
        error_est, _ = error_estimator(solution, mesh)
        error_ext = energy(solution, mesh, source_func[1], source_func[2], bc)
        eff = count_effectivity(freq, error_est, error_ext)
        prev_freq = freq
        if eff - percent > 0: 
            high = freq
            freq = (low + freq) / 2
        else:
            low = freq
            freq = (freq + high) / 2

    return freq 


def test_frequency_eff(max_iter):
    # Load models
    nn_fine = keras.models.load_model("models/Fine_NU_U_HF3GF3J2_logerr_base20_train.h5")
    nn_coarse = keras.models.load_model("models/Coarse_NU_U_HF3GF3J2_logerr_base20_train.h5")

    # Initialise AMR variables
    bc = 0
    low = 200
    high = 350
    freq = 303
    eff = 1 << 20 
    percent = 90
    i = 0 

    while abs(eff - percent) > 1 and i < max_iter:
        i += 1  
        source_func= f_str_det(freq)
        source_func_str = source_func[0]
        coarse_mesh, mesh = search_mesh_size(40, bc, source_func_str)
        error_estimator = build_nn_error_estimator(bc, source_func_str, nn_fine, nn_coarse, coarse_mesh)

        solution = solver(mesh, bc, source_func_str)
        
        error_est, _ = error_estimator(solution, mesh)
        error_ext = energy(solution, mesh, source_func[1], source_func[2], bc)
        print("freq", freq)
        eff = count_effectivity(freq, error_est, error_ext)
        print("eff", eff)
        freq += 1

    return freq 


if __name__ == '__main__':
    tolerance = 1e-2
    max_iter = 16
    file_nn = 'experiments/amr-data-nn.txt'
    file_explicit = 'experiments/amr-data-explicit.txt'
    file_recovery = 'experiments/amr-data-recovery.txt'
    for i in range(10): 
        print(f"--------------------- Iteration {i} ---------------------")
        convergence_experiments(tolerance, max_iter, file_nn, file_explicit, file_recovery)

    convergence_plots(file_nn, file_explicit, file_recovery)