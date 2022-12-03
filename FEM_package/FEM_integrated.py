# INCLUDE LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import random

# INCLUDE OWN PACKAGES
import source
import solver
import post

####### PROGRAM ########

# SOLVE PDE function -u_xx = f

# Generate mesh on which to solve
N_intervals = 10
kind = 'rand'
if kind == 'rand':
    finegrid = np.linspace(0, 1, N_intervals*3)
    xgrid = np.sort(np.random.choice(finegrid[10:], size=N_intervals, replace=False))
    xgrid = np.append(xgrid, 1)
    xgrid = np.insert(xgrid, 0, 0)
else: 
    xgrid = np.linspace(0, 1, N_intervals)
print(xgrid)
neu = random.uniform(-5, 5)


# Generate source
trig_degree = 10
f_source = source.source(trig_degree)
f_source_func = f_source.sourceTrigRand
f_source_yvalues = f_source_func(xgrid)
f_source_coeff = f_source.coeff_arr_out()

# Solver
solver_solve = solver.solver(xgrid, f_source_coeff, f_source_func)
solver_solve.assembleFEMVec()
solver_solve.assembleRHSVec(neu)
u_sol = solver_solve.solve()

# Postprocessing
post_pos = post.post(xgrid, f_source_func, f_source_coeff, trig_degree, neu, u_sol)
xfine, u_exact = post_pos.exactSol()
energy_norm = post_pos.energy()
jump = post_pos.jump()
residual = post_pos.residual_err()

data = post_pos.gendata(400)

plt.plot(xfine, u_exact, linewidth = 5)
plt.plot(xgrid, u_sol, 'ro-', linewidth = 2, markersize = 4)
plt.show()
