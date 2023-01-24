from scipy import integrate
import keras
import numpy as np

from adaptive_mesh_refinement import adaptive_mesh_refinement, f_str, build_nn_error_estimator
from plot_utilities import process_amr_data, plot_mse_error_estimate, plot_number_elements, plot_average_iterations, write_amr_data, plot_refinement


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
    u_ex_transform = Recovery_err.func(xi*(mesh[1:]-mesh[:-1]) + mesh[:-1])
    u_transform = (sol[1:] - sol[:-1]) / (mesh[1:]-mesh[:-1])
    return (u_ex_transform-u_transform)**2 * (mesh[1:]-mesh[:-1]) 


def classical_error_estimator(sol, mesh):
    Recovery_err = Recovery(sol, mesh)
    energy_squared = integrate.quad_vec(error_recovery, 0, 1, args=(sol, mesh, Recovery_err))[0]
    energy_norm = np.sqrt(energy_squared)
    global_error = np.linalg.norm(energy_norm)
    return energy_norm, global_error


def convergence_plots(nn_file, rec_file):
    data_nn, avg_run_nn =  process_amr_data(nn_file)
    data_rec, avg_run_rec = process_amr_data(rec_file)
    plot_mse_error_estimate(data_nn, data_rec)
    plot_number_elements(data_nn, data_rec)
    plot_average_iterations(avg_run_nn, avg_run_rec)


def convergence_experiments(tolerance, max_iter):
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
    tolerance = 1e-3
    max_iter = 20
    for i in range(20): 
        print(f"--------------------- Iteration {i} ---------------------")
        convergence_experiments(tolerance, max_iter)
    convergence_plots('experiments/amr-data-nn.txt', 'experiments/amr-data-rec.txt')
