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
def main(mesh, N=5, deg=3):
    N_intervals = N + 1
    kind = 'None'
    if kind == 'rand':
        finegrid = np.linspace(0, 1, N_intervals*3)
        xgrid = np.sort(np.random.choice(finegrid[10:], size=N_intervals, replace=False))
        xgrid = np.append(xgrid, 1)
        xgrid = np.insert(xgrid, 0, 0)
    else: 
        xgrid = np.linspace(0, 1, N_intervals)
    xgrid = mesh
    neu = random.uniform(-5, 5)



    # Generate source
    trig_degree = deg
    f_source = source.source(trig_degree)         #Create source
    f_source_func = f_source.sourceTrigRand    #Store source function
    f_source_yvalues = f_source_func(xgrid)       # Store source yvalues at nodes(Only needed for plot)
    f_source_coeff = f_source.coeff_arr_out()     # Store coefficients of source

    # Solver
    solver_solve = solver.solver(xgrid, f_source_coeff, f_source_func)  # Setup solver
    solver_solve.assembleFEMVec()                                       # Assemble matrix A
    solver_solve.assembleRHSVec(neu)                                    # Assemble RHS
    u_sol = solver_solve.solve()                                        # Solve with boundary conditions

    # Postprocessing
    post_pos_new = post.post(xgrid, f_source_func, f_source_coeff, trig_degree, neu, u_sol) # Setup postprocessing
    # xfine, u_exact = post_pos.exactSol()                                                # Evaluates exact solution
    energy_norm = post_pos_new.energy()
    # jump = post_pos.jump()                                                              # Jump per element
    # residual = post_pos.residual_err()                                                  # Residual per element
    x_old = np.linspace(0, 1, 40)
    solver_solve_old = solver.solver(x_old, f_source_coeff, f_source_func)  # Setup solver
    solver_solve_old.assembleFEMVec()                                       # Assemble matrix A
    solver_solve_old.assembleRHSVec(neu)                                    # Assemble RHS
    u_sol_old = solver_solve_old.solve()   
    # data = post_pos.gendata2(1)
    # print(data)

    #If you want to plot
    # plt.plot(xfine, u_exact, linewidth = 6)
    # plt.plot(xgrid, u_sol, 'ro-')
    # plt.plot(x_old, u_sol_old, 'go-')
    # plt.grid()
    # plt.show()




    
    
    # x = np.linspace(0, 1, 1000)
    # plt.plot(x, post_pos.gradTrigError(x))
    # plt.show()
    # err_rec = post_pos.energyRecovery()
    # with open("test_dataFEM.csv", "ab") as f:
    #     np.savetxt(f, data, delimiter=",")
    # data_FEM = post_pos.gendata3(1, [0.25, 0.5, 0.75])
    
    # with open("data_x_100.csv", "ab") as f:
    #     np.savetxt(f, xgrid, delimiter=",")
    
    # with open("data_grad_100.csv", "ab") as f:
    #     np.savetxt(f, post_pos.grad_fem, delimiter=",")

        # np.savetxt(f, post_pos.grad_fem, delimiter=',')
    # print(energy_norm)
    # plt.plot(post_pos.energyRecovery())
    # plt.show()


    return u_sol, xgrid, u_sol_old, x_old, energy_norm
# np.random.seed(4)
## Data Generation

# main(np.linspace(0, 1, 100001), 5, 10)
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
            data = result.gendata2(sampling_freq[i])                                                        # Generates data with frequency sampling
            with open("Gen2_coord_new_features.csv", "ab") as f:
                np.savetxt(f, data, delimiter=",")
            count += 1
            print(f"Progress {count}/{total}")
