import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (2.0, 1.0)), n=(2, 2),
                            cell_type=mesh.CellType.triangle,)

print("here comes mesh ", dir(msh))
print("geometry ", dir(msh.geometry))
print("msh.geometry.x ", msh.geometry.x)
print("msh.geometry.index_map ", dir(msh.geometry.index_map))
print("msh.geometry.index_global_indices ", msh.geometry.input_global_indices)


V = fem.FunctionSpace(msh, ("Lagrange", 1))
facets = mesh.locate_entities_boundary(msh, dim=1,
                                       marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                      np.isclose(x[0], 2.0)))
print("facets ",facets)
dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
print("dofs ", dofs)
bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds

problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()


# print('here starts printing')
# print(uh.x.array)
# msh_refined = mesh.refine(msh)
# print("here comes mesh ", dir(msh_refined))
# print("geometry ", dir(msh_refined.geometry))
# print("msh.geometry.x ", msh_refined.geometry.x)
# print("msh.geometry.index_map ", dir(msh_refined.geometry.index_map))
# print("msh.geometry.index_global_indices ", msh_refined.geometry.input_global_indices)
# facets_refined = mesh.locate_entities_boundary(msh_refined, dim=1,
#                                        marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
#                                                                       np.isclose(x[0], 2.0)))
# dofs_refined = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets_refined)
msh_refined = mesh.refine(msh, [1])
print("here comes  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ", dir(msh_refined))
print("geometry ", dir(msh_refined.geometry))
print("msh.geometry.x ", msh_refined.geometry.x)
print("msh.geometry.index_map ", dir(msh_refined.geometry.index_map))
print("msh.geometry.index_global_indices ", msh_refined.geometry.input_global_indices)
import pyvista
pyvista.start_xvfb()
topology, cell_types, geometry = plot.create_vtk_mesh(msh_refined)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
plotter.save_graphic("msh_acc_refined.pdf")
# try:
#     import pyvista
#     pyvista.start_xvfb()
#     cells, types, x = plot.create_vtk_mesh(V)
#     grid = pyvista.UnstructuredGrid(cells, types, x)
#     grid.point_data["u"] = uh.x.array.real
#     grid.set_active_scalars("u")
#     plotter = pyvista.Plotter()
#     plotter.add_mesh(grid, show_edges=True)
#     warped = grid.warp_by_scalar()
#     plotter.add_mesh(warped)
#     plotter.save_graphic("img.pdf") 
# except ModuleNotFoundError:
#     print("'pyvista' is required to visualise the solution")
#     print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")