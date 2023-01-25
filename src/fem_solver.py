from petsc4py.PETSc import ScalarType
from mpi4py import MPI
import ufl
from dolfinx import fem, io, mesh, plot
import numpy as np
from ufl import ds, dx, grad, inner, sin, cos



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