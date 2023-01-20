from numpy import sqrt, arange, round
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
from scipy.stats import gaussian_kde
import ufl
from scipy import integrate
import keras
from dolfinx import fem, io, mesh, plot
import numpy as np
from ufl import ds, dx, grad, inner, sin, cos
import matplotlib.pyplot as plt
from functools import partial

EPSILON = 1e-4


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


def get_error_estimate(features, nn_fine, nn_coarse):
    threshold = 2 ** -12
    
    # Set criteria for element to be fine
    fine_separation = np.logical_and(
        features[:, 0] < threshold, features[:, 1] < threshold, features[:, 2] < threshold)

    # Check for prescence of fine elements
    if len(features[fine_separation]) == 0:
        fine_error = 0
    else:
        fine_error = np.exp(nn_fine.predict(
            features[fine_separation][:, 3:], verbose=0).flatten())

    # Check for prescence of coarse elements
    if len(features[~fine_separation]) == 0:
        coarse_error = 0
    else:
        coarse_error = np.exp(nn_coarse.predict(
            features[~fine_separation][:, 3:], verbose=0).flatten())

    # print(f"Coarse: {(np.count_nonzero(~fine_separation) / len(features)) * 100:.4f}%, \
    #     Fine: {(np.count_nonzero(fine_separation) / len(features)) * 100:.4f}%")

    # Combine the fine and coarse errors
    local_errors = np.zeros(len(features))
    local_errors[fine_separation] = fine_error
    local_errors[~fine_separation] = coarse_error
    local_errors = np.insert(local_errors, 0, local_errors[0])
    local_errors = np.append(local_errors, local_errors[-1])

    global_error = np.linalg.norm(np.array(local_errors))
    return local_errors, global_error


def relative_change(grad_new, grad_old):
    grad_old = np.maximum(1e-9 * np.ones(len(grad_old)), grad_old)
    delta = np.abs(((grad_new - grad_old) / grad_old) * 100)
    return {
        "i-1": delta[0:-2],
        "i": delta[1:-1],
        "i+1": delta[2:],
    }


def generate_data(new_sol, new_grid, old_sol, old_grid):
    # Step size: x_i+1 - x_i
    old_step = old_grid[1:] - old_grid[:-1]
    new_step = new_grid[1:] - new_grid[:-1]

    hs = {"i-1": new_step[0:-2], "i": new_step[1:-1], "i+1": new_step[2:]}

    # Gradient: y_i+1 - y_i / x_i+1 - x_i
    old_grad = (old_sol[1:] - old_sol[:-1]) / old_step
    new_grad = (new_sol[1:] - new_sol[:-1]) / new_step

    # Search for index of left element to compare with
    index_grad = np.searchsorted(old_grid, new_grid, 'left')[1:]
    # Extend coarse grid for comparison
    inter_old_grad = np.array(list(map(lambda x: old_grad[x - 1], index_grad)))
    inter_old_grid = np.array(list(map(lambda x: old_step[x - 1], index_grad)))

    # Compute relative change in gradient and step-size
    delta_step = relative_change(new_step, inter_old_grid)
    delta_grad = relative_change(new_grad, inter_old_grad)
    delta_grad = {k: np.log(v) for k, v in delta_grad.items()}

    # Compute jump in gradient
    grad = {"i-1": new_grad[0:-2], "i": new_grad[1:-1], "i+1": new_grad[2:]}
    grad_jump_left = np.log(np.maximum(
        1e-9 * np.ones(len(grad["i"])), np.abs(grad["i-1"] - grad["i"])))
    grad_jump_right = np.log(np.maximum(
        1e-9 * np.ones(len(grad["i"])), np.abs(grad["i"] - grad["i+1"])))

    # Appends all quantities
    return np.vstack((hs["i-1"], hs["i"], hs["i+1"], delta_step["i-1"], delta_step["i"], delta_step["i+1"],
                      delta_grad["i-1"], delta_grad["i"], delta_grad["i+1"],
                      grad_jump_left, grad_jump_right)).transpose()


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


def exact_gradient(x, a_k, freq, bc_1):
    result = bc_1
    for i in range(len(a_k)):
        result += (a_k[i] / freq[i]) * np.cos(freq[i] * x)
    return result


def exact_sol(x, a_k, freq, bc_1):
    result = bc_1 * x
    for i in range(len(a_k)):
        result += (a_k[i] / freq[i] ** 2) * np.sin(freq[i] * x)
    return result


def error_master(xi, sol, mesh, a_k, freq, bc_1):
    u_ex_transform = exact_gradient(
        xi * (mesh[1:] - mesh[:-1]) + mesh[:-1], a_k, freq, bc_1)
    u_transform = (sol[1:] - sol[:-1]) / (mesh[1:] - mesh[:-1])
    return (u_ex_transform - u_transform) ** 2 * (mesh[1:] - mesh[:-1])


def energy(sol, mesh, a_k, freq, bc_1):
    energy_squared = integrate.quad_vec(
        error_master, 0, 1, args=(sol, mesh, a_k, freq, bc_1))[0]
    energy_norm = np.sqrt(energy_squared)
    return energy_norm


def solver(mesh_new, dirichletBC, f_str):
    # Define domain
    x_start = 0.0
    x_end = 1.0

    # Create mesh
    N = len(mesh_new)

    cell = ufl.Cell("interval", geometric_dimension=2)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))

    x = np.stack((mesh_new, np.zeros(N)), axis=1)
    cells = np.stack((np.arange(N - 1), np.arange(1, N)),
                     axis=1).astype("int32")
    msh = mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)

    # Define functionspace
    V = fem.FunctionSpace(msh, ("Lagrange", 1))

    # Define boundary conditions
    facets = mesh.locate_entities_boundary(msh, dim=0, marker=lambda x: np.logical_or(np.isclose(x[0], x_start),
                                                                                      np.isclose(x[0], x_end)))

    dofs1 = fem.locate_dofs_topological(V=V, entity_dim=0, entities=facets[0])
    dofs2 = fem.locate_dofs_topological(V=V, entity_dim=0, entities=facets[1])

    bc1 = fem.dirichletbc(value=ScalarType(0), dofs=dofs1, V=V)
    bc2 = fem.dirichletbc(value=ScalarType(dirichletBC), dofs=dofs2, V=V)

    # Define trial and test functions and assign coördinates on mesh
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)

    # Call and evaluate source function over domain
    f = eval(f_str)

    # Define problem
    a = (inner(grad(u), grad(v))) * dx
    L = inner(f, v) * dx

    # Solve problem
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc1, bc2], petsc_options={
                                      "ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    return uh.x.array


def refine(mesh, err_pred, global_error):
    # base case
    if len(mesh) == 1:
        return mesh

    num_elements = len(mesh)
    def compute_err(err): return err * sqrt(num_elements) / global_error

    refined_mesh = [mesh[0]]
    for i in range(0, num_elements - 1):
        curErr = compute_err(err_pred[i])
        num_points = int(round(curErr))

        refined_mesh.extend(
            mesh[i] + (mesh[i + 1] - mesh[i]) /
            (num_points + 1) * arange(1, num_points + 2)
        )

        if np.isclose(curErr, 0.5, atol=EPSILON) and \
                (i + 1 < len(err_pred) and np.isclose(compute_err(err_pred[i + 1]), 0.5, atol=EPSILON)):
            refined_mesh.pop()
    return np.array(refined_mesh)


def build_nn_error_estimator(bc, source_func_str, nn_fine, nn_coarse):
    mesh_coarse = np.linspace(0, 1, 40)
    solution_coarse = solver(mesh_coarse, bc, source_func_str)
    generate_data_partial = partial(generate_data, old_sol = solution_coarse, old_grid = mesh_coarse)
    get_error_estimate_partial = partial(get_error_estimate, nn_fine = nn_fine, nn_coarse = nn_coarse)
    return lambda solution, mesh: get_error_estimate_partial(generate_data_partial(solution, mesh))


def adaptive_mesh_refinement(tolerance, max_iter, bc, source_func_temp, error_estimator):
    # Initialise AMR variables
    mesh = np.linspace(0, 1, 60)
    x = np.linspace(0, 1, 100000)
    error = 1 << 20
    N_elements = []

    source_func_str = source_func_temp[0]
    solution_exact = exact_sol(x,  source_func_temp[1], source_func_temp[2], bc)

    # Initialise data structures for plotting
    meshes = dict()
    solutions = dict()
    est_global_errors = []
    ex_global_errors = []

    iter = 0
    while error > tolerance and iter < max_iter:
        iter += 1
        N_elements.append(len(mesh) - 1)
        solution = solver(mesh, bc, source_func_str)

        # Estimated error
        local_error, error = error_estimator(solution, mesh)
        
        # Exact error
        ex_error = energy(solution, mesh, source_func_temp[1], source_func_temp[2], bc)
        ex_global_error = np.sqrt(np.sum(ex_error ** 2))
        # print("ERROR", error)
    	
        # Save AMR information for plotting
        meshes[iter] = mesh
        solutions[iter] = solution
        est_global_errors.append(error)
        ex_global_errors.append(ex_global_error)

        mesh = refine(mesh, local_error, tolerance)

    return x, solution_exact, meshes, solutions, est_global_errors, ex_global_errors, N_elements


def plot_refinement(x, exact, meshes, solutions, est_global_errors, ex_global_errors, N_elements, plot_title):
    color = True
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(211)
    plt.plot(x, exact, lw=6, zorder=0)
    plt.grid()
    
    iter = 0
    for i, mesh in meshes.items(): 
        ax2 = plt.subplot(212)
        
        plt.title(f"Iteration: {i}, Estimated Error: {est_global_errors[i - 1]:.4e}, Exact Error: {ex_global_errors[i - 1]:.4e}")
        
        if color:
            xy = np.vstack([mesh, solutions[i]])
            z = gaussian_kde(xy)(xy)
            ax2.scatter(mesh, i * np.ones(len(mesh)), c=z, cmap='rainbow')
        else:
            ax2.scatter(mesh, i * np.ones(len(mesh)))
        iter += 1
    
    if color:
        plt.colorbar(ax2.collections[0], location='bottom', shrink=0.7)
    
    plt.grid()
    ax1.scatter(meshes[iter], solutions[iter], s=5, color='r')
    plt.gca().set_yticks(np.arange(1, iter + 1))
    plt.gca().set_yticklabels([f"Iter: {i + 1}, Elements: {N_elements[i]}" for i in range(iter)])
    plt.tight_layout()
    title = f"experiments/refinement_{plot_title}.png"
    plt.savefig(title)


def plot_error_convergence(ex_global_errors, est_global_errors_nn, est_global_errors_rec, plot_title):
    iterations = len(ex_global_errors)
    plt.figure()
    plt.semilogy(range(iterations), ex_global_errors, 'o-', color='orange', label='Exact Errors')
    plt.semilogy(range(iterations), est_global_errors_nn, 'o--', color='blue', label='Estimated Errors (NN)')
    plt.semilogy(range(iterations), est_global_errors_rec, 'o--', color='green', label='Estimated Errors (Recovery)')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Global Error')
    plt.title(plot_title)
    plt.savefig('experiments/error_plot.png')


def write_amr_data(filename, nr_elements, est_global_errors, ex_global_errors):
    f = open(filename, "a")
    for i in range(len(est_global_errors)):
        elements = str(nr_elements[i])
        est_err = str(est_global_errors[i])
        ex_err = str(ex_global_errors[i])
        iter = str(i+1)
        f.write(f"{iter}, {elements}, {est_err}, {ex_err}")
        f.write("\n")
    f.close()


def process_amr_data(filename):
    f = open(filename, "r")
    rows = f.read().splitlines()[1:-1]
    data = dict()
    iterations = []
    last = 0
    for entry in rows:
        entry = entry.split(", ")

        iteration = int(entry[0])       # 0 = iteration;
        elements = int(entry[1])        # 1 = num elements;
        global_err = float(entry[2])    # 2 = est global error;
        exact_err = float(entry[3])     # 3 = exact global error

        if not entry[0] in data: 
            data[entry[0]] = []
        
        data[entry[0]].append([elements, global_err, exact_err])

        if (iteration < last):
            iterations.append(last)
        last = iteration

    return data, np.mean(iterations)


def compute_mse_error_estimate(data):    
    mses = []
    for it, values in data.items(): 
        square_dif = list(map(lambda value: (value[1] - value[2]) * (value[1] - value[2]), values))
        mse = np.sqrt(np.sum(square_dif) / len(values))
        mses.append(mse)
    return mses 


def compute_average_elements(data):    
    num_elements = []
    for it, values in data.items(): 
        values = np.array(values)
        n_elements = values[:, 0]
        mean_elements = np.mean(n_elements) 
        num_elements.append(mean_elements)
    return num_elements


def plot_mse_error_estimate(data_nn, data_rec):
    mse_nn = compute_mse_error_estimate(data_nn)
    mse_rec = compute_mse_error_estimate(data_rec)
    plt.figure()
    plt.semilogy(range(1, len(mse_nn) + 1), mse_nn, 'o-', label='Neural Network')
    plt.semilogy(range(1, len(mse_rec) + 1), mse_rec, 'o-', label='Recovery Method')
    plt.xticks(range(1, len(mse_nn) + 1))
    plt.xlabel('Iteration')
    plt.ylabel('Error in Error Estimation')
    plt.legend()
    plt.savefig('experiments/mse.svg')


def plot_number_elements(data_nn, data_rec):
    elements_nn = compute_average_elements(data_nn)
    elements_rec = compute_average_elements(data_rec)
    plt.figure()
    relative_change = (np.array(elements_rec) / np.array(elements_nn[:-1]) * 100.0) - 100.0
    plt.plot(range(1, len(elements_nn[:-1]) + 1), relative_change , 'o--', label='Saved by Neural Network')
    plt.xticks(range(1, len(elements_nn[:-1]) + 1))
    plt.xlabel('Iteration')
    plt.ylabel('Percent of Mesh Size Reduction')
    plt.legend()
    plt.savefig('experiments/num_elements.svg')


def plot_average_iterations(avg_run_nn, avg_run_rec):
    methods = ['Neural Network', 'Recovery Method']
    avg_run = [avg_run_nn, avg_run_rec]
    plt.figure()
    plt.bar(methods, height=avg_run, width=0.4, label='Average Number of Iterations')
    plt.legend()
    plt.savefig('experiments/avg_iterations.svg')


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
    source_func_temp = f_str(1000, 40, 2)
    source_func_str = source_func_temp[0]

    # AMR with neural network
    nn_error_estimator = build_nn_error_estimator(bc, source_func_str, nn_fine, nn_coarse)
    x_nn, solution_exact_nn, meshes_nn, solutions_nn, est_global_errors_nn, ex_global_errors_nn, N_elements_nn = \
         adaptive_mesh_refinement(tolerance, max_iter, bc, source_func_temp, nn_error_estimator)
    write_amr_data('experiments/amr-data-nn.txt', N_elements_nn, est_global_errors_nn, ex_global_errors_nn)

    # # AMR with classical error estimator
    x, solution_exact, meshes, solutions, est_global_errors, ex_global_errors, N_elements = \
        adaptive_mesh_refinement(tolerance, max_iter, bc, source_func_temp, classical_error_estimator)
    write_amr_data('experiments/amr-data-rec.txt', N_elements, est_global_errors, ex_global_errors)

    # plot_refinement(x_nn, solution_exact_nn, meshes_nn, solutions_nn, est_global_errors_nn, ex_global_errors_nn, N_elements_nn, 'neural network')    
    # plot_refinement(x, solution_exact, meshes, solutions, est_global_errors, ex_global_errors, N_elements, 'classical method') 
    # plot_error_convergence(ex_global_errors, est_global_errors_nn, est_global_errors, 'Convergence of Global Error')   


if __name__ == '__main__':
    tolerance = 1e-2
    max_iter = 50
    for i in range(20): 
        print(f"---------- Iteration {i} ----------")
        convergence_experiments(tolerance, max_iter)
    convergence_plots('experiments/amr-data-nn.txt', 'experiments/amr-data-rec.txt')
