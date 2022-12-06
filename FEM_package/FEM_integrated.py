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
def main(N=10, deg=10):
    N_intervals = N
    kind = 'None'
    if kind == 'rand':
        finegrid = np.linspace(0, 1, N_intervals*3)
        xgrid = np.sort(np.random.choice(finegrid[10:], size=N_intervals, replace=False))
        xgrid = np.append(xgrid, 1)
        xgrid = np.insert(xgrid, 0, 0)
    else: 
        xgrid = np.linspace(0, 1, N_intervals)

    neu = random.uniform(-5, 5)


    # Generate source
    trig_degree = deg
    f_source = source.source(trig_degree)         #Create source
    f_source_func = f_source.sourceTrigRand      #Store source function
    f_source_yvalues = f_source_func(xgrid)       # Store source yvalues at nodes(Only needed for plot)
    f_source_coeff = f_source.coeff_arr_out()     # Store coefficients of source

    # Solver
    solver_solve = solver.solver(xgrid, f_source_coeff, f_source_func)  # Setup solver
    solver_solve.assembleFEMVec()                                       # Assemble matrix A
    solver_solve.assembleRHSVec(neu)                                    # Assemble RHS
    u_sol = solver_solve.solve()                                        # Solve with boundary conditions

    # Postprocessing
    post_pos = post.post(xgrid, f_source_func, f_source_coeff, trig_degree, neu, u_sol) # Setup postprocessing
    xfine, u_exact = post_pos.exactSol()                                                # Evaluates exact solution
    energy_norm = post_pos.energy()                                                     # Energy norm per element
    jump = post_pos.jump()                                                              # Jump per element
    residual = post_pos.residual_err()                                                  # Residual per element
    x = np.linspace(0, 1, 1000)
    grad_values = post_pos.grad(x)
    grad_exact = post_pos.gradTrigError(x)

    post_pos.energyRecovery()


    #If you want to plot
    plt.plot(x, grad_exact)
    plt.plot(x, grad_values)
    plt.plot(xfine, u_exact, linewidth = 5)
    plt.plot(xgrid, u_sol, 'ro-', linewidth = 2, markersize = 4)
    plt.show()
    data = post_pos.gendata(1, [0.25, 0.5, 0.75])
    print(data)
    return post_pos

main(N=11)
## Data Generation
if False:
    N_samples = 3000
    base_source = 10
    patch_size = 3 # including own element

    elements = [2**i for i in range(3, 18)]
    for i in elements.copy():
        if i < (patch_size):
            elements.remove(i)

    sampling_freq = []
    funcs = []
    for i in elements:
        if (N_samples / (base_source * (i - (patch_size-1)))) <= 1:
            freq = int(i / (N_samples/base_source + (patch_size-1)))
            sampling_freq.append(freq) 
            funcs.append(base_source)      
            
        else:
            sampling_freq.append(1)
            funcs.append(int(N_samples / i))

    total = np.sum(funcs)
    count = 0
    for i, N in enumerate(elements):
        for j in range(funcs[i]):
            result = main(N, 10)
            data = result.gendata(sampling_freq[i], [0.2, 0.5, 0.8])                                                        # Generates data with frequency sampling
            with open("13_new_features.csv", "ab") as f:
                np.savetxt(f, data, delimiter=",")
            count += 1
            print(f"Progress {count}/{total}")
