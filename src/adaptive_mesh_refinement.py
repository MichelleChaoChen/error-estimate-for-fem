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

EPSILON = 1e-4

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

    print(f"Coarse: {(np.count_nonzero(~fine_separation) / len(features)) * 100:.4f}%, \
        Fine: {(np.count_nonzero(fine_separation) / len(features)) * 100:.4f}%")

    # Combine the fine and coarse errors
    local_errors = np.zeros(len(features))
    local_errors[fine_separation] = fine_error
    local_errors[~fine_separation] = coarse_error

    global_error = np.linalg.norm(np.array(local_errors))
    return local_errors, global_error


def relative_change(grad_new, grad_old):
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

    # Extend boundary on the old mesh
    virtual_point_old_left = (old_sol[0] - (old_sol[1] - old_sol[0]))
    old_sol = np.insert(old_sol, 0, virtual_point_old_left)
    old_grid = np.insert(old_grid, 0, old_grid[0] - old_step[0])
    virtual_point_old_right = (old_sol[-1] + old_sol[-1] - old_sol[-2])
    old_sol = np.append(old_sol, virtual_point_old_right)
    old_grid = np.append(old_grid, old_grid[-1] + old_step[-1])
    old_step = old_grid[1:] - old_grid[:-1]

    # Extend boundary on the new mesh
    virtual_point_new_left = (new_sol[0] - (new_sol[1] - new_sol[0]))
    new_sol = np.insert(new_sol, 0, virtual_point_new_left)
    new_grid = np.insert(new_grid, 0, new_grid[0] - new_step[0])
    virtual_point_new_right = (new_sol[-1] - (new_sol[-1] - new_sol[-2]))
    new_sol = np.append(new_sol, virtual_point_new_right)
    new_grid = np.append(new_grid, new_grid[-1] + new_step[-1])
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


def plot_refinement(x, exact, meshes, solutions, est_global_errors, ex_global_errors, N_elements, color=True):
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
    plt.savefig(f"refinement_plot")        



def adaptive_mesh_refinement(tolerance, max_iter):
    mesh_coarse = np.linspace(0, 1, 40)
    mesh = np.linspace(0, 1, 60)
    x = np.linspace(0, 1, 100000)
    bc = 0
    error = 1 << 20
    N_elements = []

    source_func_temp = f_str(1000, 30, 3)
    source_func_str = source_func_temp[0]

    solution_coarse = solver(mesh_coarse, bc, source_func_str)
    solution_exact = exact_sol(x,  source_func_temp[1], source_func_temp[2], bc)

    # Load models
    nn_fine = keras.models.load_model("models/Fine_NU_U_HF3GF3J2_logerr_base20_train.h5")
    nn_coarse = keras.models.load_model("models/Coarse_NU_U_HF3GF3J2_logerr_base20_train.h5")

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

        features = generate_data(solution, mesh, solution_coarse, mesh_coarse)
        local_error, error = get_error_estimate(features, nn_fine, nn_coarse)
        ex_error = energy(solution, mesh, source_func_temp[1], source_func_temp[2], bc)
        ex_global_error = np.sqrt(np.sum(ex_error ** 2))
        print("ERROR", error)
    	
        # Save AMR information for plotting
        meshes[iter] = mesh
        solutions[iter] = solution
        est_global_errors.append(error)
        ex_global_errors.append(ex_global_error)

        mesh = refine(mesh, local_error, error)

    # Plot the AMR process
    plot_refinement(x, solution_exact, meshes, solutions, est_global_errors, ex_global_errors, N_elements)    
    return solution


if __name__ == '__main__':
    tolerance = 1e-2
    max_iter = 100
    adaptive_mesh_refinement(tolerance, max_iter)