from numpy import sqrt, arange, round
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
import ufl
from scipy import integrate
import keras
from dolfinx import fem, io, mesh, plot
import numpy as np
from ufl import ds, dx, grad, inner, sin, cos
from functools import partial
from plot_utilities import plot_refinement

EPSILON = 1e-4

def get_error_estimate(features, nn_fine, nn_coarse):
    """
    Provides the estimate of the local error and global error
    of feature vectors from one iteration of adaptive mesh refinement. 

    :param features: Feature vectors for every element on current mesh
    :param nn_fine: Neural network for fine elements
    :param nn_coarse: Neural network for coarse elements
    :return: Array of local error on each element and global error
    """
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

    print(f"Coarse: {(np.count_nonzero(~fine_separation) / len(features)) * 100:.4f}%, \
        Fine: {(np.count_nonzero(fine_separation) / len(features)) * 100:.4f}%")

    # Combine the fine and coarse errors
    local_errors = np.zeros(len(features))
    local_errors[fine_separation] = fine_error
    local_errors[~fine_separation] = coarse_error

    global_error = np.linalg.norm(np.array(local_errors))
    return local_errors, global_error


def relative_change(grad_new, grad_old):
    """
    Computes relative change between a batch of quantities. 

    :param grad_new: Quantities from most recent FEM solution
    :param grad_old: Quantities from base FEM solution
    :return: 
    """
    delta = np.abs(((grad_new - grad_old) / grad_old) * 100)
    return {
        "i-1": delta[0:-2],
        "i": delta[1:-1],
        "i+1": delta[2:],
    }


def extend_boundary(sol, grid, step):
    """
    Extends the boundaries on a mesh using virtual points. 

    :param sol: Computed solution
    :param grid: The grid to extend the boudary on
    :param step: Step sizes of the grid
    :return: Solution, grid, and stepsize with virtual points on the boundary
    """
    virtual_point_left = (sol[0] - (sol[1] - sol[0]))
    sol = np.insert(sol, 0, virtual_point_left)
    grid = np.insert(grid, 0, grid[0] - step[0])
    virtual_point_right = (sol[-1] + sol[-1] - sol[-2])

    sol = np.append(sol, virtual_point_right)
    grid = np.append(grid, grid[-1] + step[-1])
    step = grid[1:] - grid[:-1]

    return sol, grid, step


def generate_data(new_sol, new_grid, old_sol, old_grid):
    """
    Creates feature vectors for all elements of new FEM solution
    using both the new solution and base solution. 

    :param new_sol: New FEM solution
    :param new_grid: Mesh on which the new FEM solution was computed
    :param old_sol: Base FEM solution
    :param old_grid: Coarse mesh for base solution
    :return: Feature vector used to predict error of new FEM solution 
    """
    # Step size: x_i+1 - x_i
    old_step = old_grid[1:] - old_grid[:-1]
    new_step = new_grid[1:] - new_grid[:-1]

    # Extend boundary on the old and new mesh
    old_sol, old_grid, old_step = extend_boundary(old_sol, old_grid, old_step)
    new_sol, new_grid, new_step = extend_boundary(new_sol, new_grid, new_step)

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


def f_str(coeff_range, freq_range, N):
    """
    Function that creates random source function 
    using sines. 

    :param coeff_range: Range of coefficients of sines
    :param freq_range: Range of frequency of sines
    :param N: The number of terms in the sum of sines
    :return: Source function in string format
    """
    a_k = np.random.uniform(coeff_range-1, coeff_range, N)
    freq = np.pi * np.random.randint(freq_range-1, freq_range, N)
    my_string = ''
    for i in range(N):
        string = "%s*sin(%s*x[0])" % (str(a_k[i]), str(freq[i]))
        if i != N - 1:
            string += ' + '
        my_string += string
    return [my_string, a_k, freq]


def f_exp(a_k, freq, x):
    """
    Defines the source function in non-string format
    using the coefficients and  frequency, so that the 
    exact solution can be found. 

    :param a_k: Coefficients of sines in source function
    :param freq: Frequencies of sines in source function
    :param x: Domain of the source function
    :return: Source function
    """
    f = 0
    for i in range(len(a_k)):
        f += a_k[i] * np.sin(freq[i] * x)
    return f


def exact_gradient(x, a_k, freq, bc_1):
    """
    Computes the exact gradient of the solution. 

    :param x: Domain of function
    :param a_k: Coefficients of source function
    :param freq: Frequency of source function
    :param bc_1: Boundary condition 
    :return: Exact gradient of solution
    """
    result = bc_1
    for i in range(len(a_k)):
        result += (a_k[i] / freq[i]) * np.cos(freq[i] * x)
    return result


def exact_sol(x, a_k, freq, bc_1):
    """
    Computse the exact solution. 

    :param x: Domain of function
    :param a_k: Coefficients of source function
    :param freq: Frequency of source function
    :param bc_1: Boundary condition
    :return: Value of exact solution 
    """
    result = bc_1 * x
    for i in range(len(a_k)):
        result += (a_k[i] / freq[i] ** 2) * np.sin(freq[i] * x)
    return result


def error_master(xi, sol, mesh, a_k, freq, bc_1):
    """
    Creates the function for evaluating the energy norm. 

    :param xi: Domain of function 
    :param sol: FEM solution
    :param mesh: Mesh used for FEM solution
    :param a_k: Coefficients of source function
    :param freq: Frequency of source function
    :param bc_1: Boundary condition 
    :return: Function for evaluating energy norm
    """
    u_ex_transform = exact_gradient(
        xi * (mesh[1:] - mesh[:-1]) + mesh[:-1], a_k, freq, bc_1)
    u_transform = (sol[1:] - sol[:-1]) / (mesh[1:] - mesh[:-1])
    return (u_ex_transform - u_transform) ** 2 * (mesh[1:] - mesh[:-1])


def energy(sol, mesh, a_k, freq, bc_1):
    """
    Computes the energy norm of the solution. 

    :param sol: Domain of function
    :param mesh: Mesh used for FEM solution
    :param a_k: Coefficients of source function
    :param freq: Frequency of source function
    :param bc_1: Boundary condition
    :return: Energy norm on the solution 
    """
    energy_squared = integrate.quad_vec(
        error_master, 0, 1, args=(sol, mesh, a_k, freq, bc_1))[0]
    energy_norm = np.sqrt(energy_squared)
    return energy_norm


# move to different file
def solver(mesh_new, dirichletBC, f_str):
    """
    Computes the FEM solution using FEniCS. 

    :param mesh_new: The mesh used for the solution
    :param dirichletBC: Boundary condition
    :param f_str: Source function in string format
    :return: FEM solution 
    """
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
    """
    Refines mesh based on local error estimate: 
    - An element is split into 2 equal halves if local error is high
    - Two elements are merged if the local error is too low 

    :param mesh: Mesh to refine
    :param err_pred: Local error on every element
    :param global_error: Global error threshold
    :return: Mesh refined based on local error
    """
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


def adaptive_mesh_refinement(tolerance, max_iter, bc, source_func, error_estimator):
    """
    Performs Adaptive Mesh Refinement: 
    1. Initializes parameters of the problem
    2. Computes a base solution on a coarse mesh
    3. Refines mesh until the global error falls below desired threshold
    4. Returns final solution

    :param tolerance: Global error tolerance
    :param max_iter: Maximum iterations of refinement
    :param bc: Boundary condition of problem
    :param source_func: Array of values describing the source function
    :param error_estimator: User-selected error estimator
    :return: Data from running the AMR pipeline
    """
    # Initialise AMR variables
    mesh = np.linspace(0, 1, 60)
    x = np.linspace(0, 1, 100000)
    error = 1 << 20
    N_elements = []

    source_func_str = source_func[0]
    solution_exact = exact_sol(x,  source_func[1], source_func[2], bc)

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
        ex_error = energy(solution, mesh, source_func[1], source_func[2], bc)
        ex_global_error = np.sqrt(np.sum(ex_error ** 2))
        print("ESTIMATED ERROR", error)
    	
        # Save AMR information for plotting
        meshes[iter] = mesh
        solutions[iter] = solution
        est_global_errors.append(error)
        ex_global_errors.append(ex_global_error)

        mesh = refine(mesh, local_error, error)

    return x, solution_exact, meshes, solutions, est_global_errors, ex_global_errors, N_elements


def run_adaptive_mesh_refinement(tolerance, max_iter):
    # Load models
    nn_fine = keras.models.load_model("models/Fine_NU_U_HF3GF3J2_logerr_base20_train.h5")
    nn_coarse = keras.models.load_model("models/Coarse_NU_U_HF3GF3J2_logerr_base20_train.h5")

    # Initialise AMR variables
    bc = 0
    source_func_temp = f_str(1000, 40, 2)
    source_func_str = source_func_temp[0]

    # AMR with neural network
    nn_error_estimator = build_nn_error_estimator(bc, source_func_str, nn_fine, nn_coarse)
    x, solution_exact, meshes, solutions, est_global_errors, ex_global_errors, N_elements = \
         adaptive_mesh_refinement(tolerance, max_iter, bc, source_func_temp, nn_error_estimator)

    # Plot the refinement results
    plot_refinement(x, solution_exact, meshes, solutions, est_global_errors, ex_global_errors, N_elements)
    return solutions


if __name__ == '__main__':
    tolerance = 1e-2
    max_iter = 13
    run_adaptive_mesh_refinement(tolerance, max_iter)