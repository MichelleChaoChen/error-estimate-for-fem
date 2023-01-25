from scipy import integrate
import numpy as np


def f_str(coeff_range, freq_range, N):
    """
    Function that creates random source function 
    using sines. 

    :param coeff_range: Range of coefficients of sines
    :param freq_range: Range of frequency of sines
    :param N: The number of terms in the sum of sines
    :return: Source function in string format
    """
    a_k = np.random.uniform(-coeff_range, coeff_range, N)
    freq = np.pi * np.random.randint(1, freq_range, N)
    my_string = ''
    for i in range(N):
        string = "%s*sin(%s*x[0])" % (str(a_k[i]), str(freq[i]))
        if i != N - 1:
            string += ' + '
        my_string += string
    print(my_string)
    return [my_string, a_k, freq]


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


def write_amr_data(filename, nr_elements, est_global_errors, ex_global_errors):
    """
    Writes data from AMR to file so that it can be
    processed later for plotting performance. 

    :param filename: The name of file to write to
    :param nr_elements: Array containing the mesh size
    :param est_global_errors: Estimated global errors
    :param ex_global_errors: Exact global errors
    """
    f = open(filename, "a")
    for i in range(len(est_global_errors)):
        elements = str(nr_elements[i])
        est_err = str(est_global_errors[i])
        ex_err = str(ex_global_errors[i])
        iter = str(i+1)
        f.write(f"{iter}, {elements}, {est_err}, {ex_err}")
        f.write("\n")
    f.close()


def process_amr_data(filename):
    """
    Reads data from AMR data file back and returns all  
    data in a dictionary format. 

    :param filename: The name of file to read from
    :return: Data from AMR and average number of iterations per AMR run 
    """
    f = open(filename, "r")
    rows = f.read().splitlines()[:-1]
    data = dict()
    iterations = []
    last = 0
    for entry in rows:
        entry = entry.split(", ")

        iteration = int(entry[0])       # 0 = iteration;
        elements = int(entry[1])        # 1 = num elements;
        global_err = float(entry[2])    # 2 = est global error;
        exact_err = float(entry[3])     # 3 = exact global error

        if not entry[0] in data: 
            data[entry[0]] = []
        
        data[entry[0]].append([elements, global_err, exact_err])

        if (iteration < last):
            iterations.append(last)
        last = iteration

    return data, np.mean(iterations)
