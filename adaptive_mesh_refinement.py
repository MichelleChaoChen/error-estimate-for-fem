from numpy import sqrt, arange, round

EPSILON = 1e-4
import keras
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import ufl
from scipy import integrate
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner, sin, cos
from scipy.stats import gaussian_kde
from mpi4py import MPI
from petsc4py.PETSc import ScalarType


def get_error_estimate(features):
    nn_fine = keras.models.load_model("models/Fine_NU_U_HF3GF3J2_logerr_base20_train.ckpt")
    nn_coarse = keras.models.load_model("models/Coarse_NU_U_HF3GF3J2_logerr_base20_train.ckpt")
    threshold = 2**-12
    fine_separation = np.logical_and(features[:, 0] < threshold, features[:, 1] < threshold, features[:, 2] < threshold)

    try:
        fine_error = np.exp(nn_fine.predict(features[fine_separation][:, 3:], verbose=0).flatten())

    except:
        fine_error = 0
    
    try:
        coarse_error = np.exp(nn_coarse.predict(features[~fine_separation][:, 3:], verbose=0).flatten())
    except:
        coarse_error = 0

    print(f"Coarse: {(np.count_nonzero(~fine_separation) / len(features))*100:.4f}%, \
        Fine: {(np.count_nonzero(fine_separation) / len(features))*100:.4f}%")

    local_errors = np.zeros(len(features))
    local_errors[fine_separation] = fine_error
    local_errors[~fine_separation] = coarse_error 

    local_errors = np.insert(local_errors, 0, local_errors[0])
    local_errors = np.append(local_errors, local_errors[-1]) 

    global_error = np.linalg.norm(np.array(local_errors))
    return local_errors, global_error


##########################################
# Feature generation
##########################################
def factor(grad_new, grad_old):
    return np.abs(((grad_new - grad_old) / grad_old) * 100)

def gendata(new_sol, new_grid, old_sol, old_grid):
    hs = new_grid[1:] - new_grid[:-1]
    hs_i = hs[1:-1]
    hs_m1 = hs[0:-2]
    hs_p1 = hs[2:]


    old_grad = (old_sol[1:] - old_sol[:-1])/(old_grid[1:] - old_grid[:-1]) 
    new_grad = (new_sol[1:] - new_sol[:-1])/(new_grid[1:] - new_grid[:-1]) 

    new_step = new_grid[1:] - new_grid[:-1]
    old_step = old_grid[1:] - old_grid[:-1]


    index_grad = np.searchsorted(old_grid, new_grid, 'left')[1:]
    inter_old_grad = np.array(list((map(lambda x: old_grad[x-1], index_grad))))
    inter_old_grid = np.array(list(map(lambda x: old_step[x-1], index_grad)))
    factor_grad = factor(new_grad, inter_old_grad)
    factor_step = factor(new_step, inter_old_grid)



    factor_step_im1 = factor_step[0:-2]
    factor_step_i = factor_step[1:-1]
    factor_step_ip1 = factor_step[2::]

    factor_grad_im1 = np.log(factor_grad[0:-2])
    factor_grad_i = np.log(factor_grad[1:-1])
    factor_grad_ip1 = np.log(factor_grad[2:])



    grad_im1 = new_grad[0:-2]
    grad_i = new_grad[1:-1]
    grad_ip1 =new_grad[2:]


    jump_1 = np.log(np.abs(grad_im1 - grad_i))
    jump_2 = np.log(np.abs(grad_i - grad_ip1)) 
    
    data = np.vstack((hs_m1, hs_i, hs_p1, factor_step_im1, factor_step_i, factor_step_ip1, \
                    factor_grad_im1, factor_grad_i, factor_grad_ip1, \
                    jump_1, jump_2)).transpose()

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


def exact_gradient(x, a_k, freq, bc_1):
    result = bc_1
    for i in range(len(a_k)):
        result += (a_k[i] / freq[i]) * np.cos(freq[i] * x) 
    return result

def exact_sol(x, a_k, freq, bc_1):
    result = bc_1 * x
    for i in range(len(a_k)):
        result += (a_k[i] / freq[i]**2) * np.sin(freq[i] * x) 
    return result

def error_master(xi, sol, mesh, a_k, freq, bc_1):
    u_ex_transform = exact_gradient(xi*(mesh[1:]-mesh[:-1]) + mesh[:-1], a_k, freq, bc_1)
    u_transform = (sol[1:] - sol[:-1]) / (mesh[1:]-mesh[:-1])
    return (u_ex_transform-u_transform)**2 * (mesh[1:]-mesh[:-1]) 

def energy(sol, mesh, a_k, freq, bc_1):
    energy_squard = integrate.quad_vec(error_master, 0, 1, args=(sol, mesh, a_k, freq, bc_1))[0]
    energy_norm = np.sqrt(energy_squard)
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
    cells = np.stack((np.arange(N - 1), np.arange(1, N)), axis=1).astype("int32")
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
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc1, bc2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    return uh.x.array


def refine(mesh, err_pred, global_error):
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
    return np.array(refined_mesh)


def adaptive_mesh_refinement():
    color=True  # Expensive but nice
    mesh = np.linspace(0, 1, 60)
    mesh_old = np.linspace(0, 1, 40)

    source_func_temp = f_str(1000, 40, 2)
    source_func_str = source_func_temp[0]
    source_func = partial(f_exp, source_func_temp[1], source_func_temp[2])
    bc = np.random.uniform(-10, 10)
    bc=0
    tolerance = 1e-2
    error = 1 << 20
    iter = 0

    solution_old = solver(mesh_old, bc, source_func_str)
    N_elements = []


    x = np.linspace(0, 1, 100000)
    exact = exact_sol(x,  source_func_temp[1], source_func_temp[2], bc)
    exact_grad = exact_gradient(x,  source_func_temp[1], source_func_temp[2], bc)

    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(211)
    plt.plot(x, exact, lw=6, zorder=0)
    plt.grid()
    while error > tolerance:
        iter += 1
        N_elements.append(len(mesh)-1)
        solution = solver(mesh, bc, source_func_str)

        features = gendata(solution, mesh, solution_old, mesh_old)
        local_error, error = get_error_estimate(features)
        print("ERROR", error)
        ex_error = energy(solution, mesh, source_func_temp[1], source_func_temp[2], bc)
        ex_global_error = np.sqrt(np.sum(ex_error**2))

        ax2 = plt.subplot(212)
        plt.title(f"Iteration: {iter}, Error: {error:.4e}, Exact Error: {ex_global_error:.4e}")
        
        if color:
            xy = np.vstack([mesh, solution])
            z = gaussian_kde(xy)(xy)
            ax2.scatter(mesh, iter * np.ones(len(mesh)), c=z, cmap='rainbow')
        else:
            ax2.scatter(mesh, iter * np.ones(len(mesh)))

        # if iter >= 10:
        #     mesh_cor = mesh
        #     break

        mesh_cor = mesh
        mesh = refine(mesh, local_error, error)
    
    if color:
        
        plt.colorbar(ax2.collections[0], location='bottom', shrink=0.7)
    plt.grid()
    ax1.scatter(mesh_cor, solution, s=5, color='r')
    plt.gca().set_yticks(np.arange(1, iter+1))
    plt.gca().set_yticklabels([f"Iter: {i+1}, Elements: {N_elements[i]}" for i in range(iter)])
    plt.tight_layout()
    plt.savefig(f"refinement_plot")        
    return solution

if __name__ == '__main__':
    adaptive_mesh_refinement()
