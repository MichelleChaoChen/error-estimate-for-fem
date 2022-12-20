from numpy import sqrt, arange
EPSILON = 1e-4
import keras
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import ufl

from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner, sin,cos

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

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


#Function that creates random sourcefunction
def f_str(coeff_range,freq_range,N):
    a_k = np.random.uniform(-coeff_range,coeff_range,N)
    freq = np.pi*np.random.randint(1,freq_range,N)
    my_string = ''
    for i in range(N):
        string = "%s*sin(%s*x[0])" % (str(a_k[i]),str(freq[i]))
        if i != N-1:
            string += ' + '
        my_string += string
    return [my_string,a_k,freq]


#Define source function non-string format
def f_exp(a_k,freq,x):
    f = 0
    for i in range(len(a_k)):
        f += a_k[i]*np.sin(freq[i]*x)
    return f


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
    a = inner(grad(u), grad(v)) * dx
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
    compute_err = lambda err : err * sqrt(num_elements) / global_error

    refined_mesh = [mesh[0]]
    for i in range(0, num_elements - 1):
        curErr = compute_err(err_pred[i])
        num_points = int(round(curErr))

        refined_mesh.extend(
            mesh[i] + (mesh[i+1] - mesh[i]) / (num_points + 1) * arange(1, num_points + 2)
        )

        if np.isclose(curErr, 0.5, atol=EPSILON) and \
                (i + 1 < len(err_pred) and np.isclose(compute_err(err_pred[i+1]), 0.5, atol=EPSILON)):
            refined_mesh.pop()

    return refined_mesh


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
