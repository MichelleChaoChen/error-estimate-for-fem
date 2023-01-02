# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:49:29 2022

@author: AlvaroBailletBolivar
"""
import numpy as np
from itertools import combinations
from random_mesh_generation import randomMeshGen 
import FEM_integrated
import grad_features
def get_data_nonuniform(total_data, base_source, patch_size, name_coarse, name_fine):
    '''
    

    Parameters
    ----------
    total_data : int
        The total number of data points to be generated. For example: 60K.
    base_source : int
        Minimum number of source functions to be generated per mesh.
    patch_size : int
        Size of patch (for example 3).
    name_coarse : str
        Name of coarse data file to be generated. Should end in .csv.
    name_fine : str
        Name of fine data file to be generated. Should end in .csv.

    Returns
    -------
    None, but writes the file with data.

    '''
    

    N_samples = int(total_data/base_source)  # amount of data to be generated per source function

    orders_of_2  = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    
    # The code below determines how many source functions must be generated per stepsize
    # in order to create a balanced dataset.
    
    # For all step sizes the solver will be run for at least 10 source functions, to ensure variety in the data.
    # If you want to run for a different base value you can change the base_source parameter.

    elements = []  #approximate number of elements per step-size 
    # (note that this value is approximate because the mesh is random and not a fixed stepsize)
    length_parameters = []
    i = 0
    for comb in combinations(orders_of_2, 2):
        if (comb[1] - comb[0]) <= 7: 
            # This says that the difference between the smallest and largest stepsize in the mesh
            # must be at most 7 powers of 2. 
            # For example: the mesh may consist of elements ranging from 2**-3 to 2**-10 but not 2**-3 to 2**-11.
            # Note intermediate ranges such as 2**-4 to 2**-7 are also generated.
            x = randomMeshGen(comb[1], comb[0])
            length_parameters.append((comb[1], comb[0]))
            elements.append(len(x))
            i += 1
    for i in elements.copy():
        if i < (patch_size):
            elements.remove(i)

    funcs = []
    for i in elements:
        if (N_samples / (base_source * (i - (patch_size-1)))) <= 1:
            funcs.append(base_source)            
        else:
            funcs.append(int(N_samples / i))

    total = np.sum(funcs)
    count = 0
    # print("test")
    ###############################################
    # DATA GENERATION PIPELINE 
    ###############################################
    for i, N in enumerate(elements):
        low_h, high_h = length_parameters[i]
        for j in range(funcs[i]):
            mesh = randomMeshGen(low_h, high_h) #This function generates a random mesh with stepsizes ranging from [2**-high_h, 2**-low_h]
            ############################################################################

            # CALL FEM SOLVER
            # (example)
            u_sol, xgrid, u_sol_old, x_old, energy_norm = FEM_integrated.main(mesh, 10, 10)
            # GENERATE DATA USING OWN FUNCTION
            # (example)
            data = grad_features.gendata(u_sol, xgrid, u_sol_old, x_old, energy_norm, 1).transpose()
            #############################################################################
            if low_h < 12 and high_h < 12:
                with open(name_coarse, "ab") as f:
                    np.savetxt(f, data, delimiter=",")
            else:
                data = data[::20]
                with open(name_fine, "ab") as f:
                    np.savetxt(f, data, delimiter=",")
            count += 1
            print(f"Progress: {count}/{total}")
            
get_data_nonuniform(100000, 10, 3, "Coarse_factor_test.csv", "Fine_factor_test_base20_test.csv")

def get_data_uniform(total_data, name, base_source=10, patch_size=3, coarse=True):
    '''
    

    Parameters
    ----------
    total_data : int
        The total number of data points to be generated. For example: 60K.
    base_source : int
        Minimum number of source functions to be generated per mesh. Recommended value is 10
    patch_size : int
        Size of patch (for example 3).
    name : str
        Name of data file to be generated. Should end in .csv..
    coarse : TYPE, boolean
        DESCRIPTION. Whether to generate a coarse data set. The default is True. Set to False to generate a fine dataset.

    Returns
    -------
    None, but writes datafile.

    '''
    N_samples = int(total_data/base_source) 
    
    if coarse:
        elements = [2**i for i in range(6, 13)] 
    else:
        elements = [2**i for i in range(13, 18)]


    for i in elements.copy():
        if i < (patch_size):
            elements.remove(i)
    
    
    # The code below determines how many source functions must be generated per stepsize
    # in order to create a balanced dataset.
    
    # For all step sizes the solver will be run for at least 10 source functions, to ensure variety in the data.
    # If you want to run for a different base value you can change the base_source parameter.
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
    
    ###############################################
    # DATA GENERATION PIPELINE 
    ###############################################
    total = np.sum(funcs)
    count = 0
    for i, N in enumerate(elements):
        for j in range(funcs[i]):
            ############################################################################
            # CALL FEM SOLVER
            # (example)
            mesh = np.linspace(0, 1, N)
            u_sol, xgrid, u_sol_old, x_old, energy_norm = FEM_integrated.main(mesh, 10, 10)
            # GENERATE DATA USING OWN FUNCTION
            # (example)
            data = grad_features.gendata(u_sol, xgrid, u_sol_old, x_old, energy_norm, sampling_freq[i]).transpose()

            #############################################################################                                                    # Generates data with frequency sampling
            with open(name, "ab") as f:
                np.savetxt(f, data, delimiter=",")
            count += 1
            print(f"Progress: {count}/{total}")

get_data_uniform(150000, "Fine_U_HF3GF3J2_logerr_base20_test.csv", base_source=10, patch_size=3, coarse=False)
get_data_uniform(50000, "Coarse_U_HF3GF3J2_logerr_base20_test.csv", base_source=10, patch_size=3, coarse=True)
