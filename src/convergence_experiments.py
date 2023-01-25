import keras

from adaptive_mesh_refinement import adaptive_mesh_refinement, f_str, build_nn_error_estimator
from plot_utilities import plot_mse_error_estimate, plot_number_elements, plot_average_iterations, plot_refinement
from utils import process_amr_data, write_amr_data
from recovery_method import classical_error_estimator


def convergence_plots(nn_file, rec_file):
    """
    Creates several plots that demonstrate the convergence of 
    adaptive mesh refinement using neural network as error estimator 
    and compares the performance with recovery-based estimator:
    1. Comparison of the MSE in error estimation of neural network and recovery method
    2. Comparison of the average mesh size over iterations
    3. Comparsion of the average number of iterations refinement needed

    :param nn_file: File containing AMR data with neural network
    :param rec_file: File containing AMR data with recovery-based error estimator
    """
    data_nn, avg_run_nn =  process_amr_data(nn_file)
    data_rec, avg_run_rec = process_amr_data(rec_file)
    plot_mse_error_estimate(data_nn, data_rec)
    plot_number_elements(data_nn, data_rec)
    plot_average_iterations(avg_run_nn, avg_run_rec)


def convergence_experiments(tolerance, max_iter):
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

    # AMR with neural network
    nn_error_estimator = build_nn_error_estimator(bc, source_func_str, nn_fine, nn_coarse)
    x_nn, solution_exact_nn, meshes_nn, solutions_nn, est_global_errors_nn, ex_global_errors_nn, N_elements_nn = \
         adaptive_mesh_refinement(tolerance, max_iter, bc, source_func, nn_error_estimator)
    write_amr_data('experiments/amr-data-nn.txt', N_elements_nn, est_global_errors_nn, ex_global_errors_nn)

    # AMR with classical error estimator
    x, solution_exact, meshes, solutions, est_global_errors, ex_global_errors, N_elements = \
        adaptive_mesh_refinement(tolerance, max_iter, bc, source_func, classical_error_estimator)
    write_amr_data('experiments/amr-data-rec.txt', N_elements, est_global_errors, ex_global_errors)

    plot_refinement(x_nn, solution_exact_nn, meshes_nn, solutions_nn, est_global_errors_nn, ex_global_errors_nn, N_elements_nn, 'neural network')    
    plot_refinement(x, solution_exact, meshes, solutions, est_global_errors, ex_global_errors, N_elements, 'classical method') 


if __name__ == '__main__':
    tolerance = 1e-2
    max_iter = 20
    # for i in range(10): 
    #     print(f"--------------------- Iteration {i} ---------------------")
    #     convergence_experiments(tolerance, max_iter)
    # could be changed to pass the text file as argument
    convergence_plots('experiments/amr-data-nn.txt', 'experiments/amr-data-rec.txt')
