from scipy import integrate
import numpy as np
from fem_solver import solver
from functools import partial


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
    return [my_string, a_k, freq]


def f_str_det(freq):
    freq = np.pi * freq
    my_string = ''
    string = "%s*sin(%s*x[0])" % (str(freq), str(freq))
    my_string += string
    return [my_string, [freq], [freq]]


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


def process_training_data(filename):
    f = open(filename, "r")
    rows = f.read().splitlines()
    train_loss = []
    validation_loss = []
    for entry in rows:
        entry = entry.split(", ")

        loss = float(entry[0])
        val_loss = float(entry[1])
        
        train_loss.append(loss)
        validation_loss.append(val_loss)

    return train_loss, validation_loss


def search_mesh_size(B, bc, f_source_str):
    B = int((B / 4)) * 4

    p = compute_p(B, bc, f_source_str)
    while p < 0.9 or p > 1.1:
        B *= 2 
        p = compute_p(B, bc, f_source_str)
    
    I = int(1.5 * B) 
    return np.linspace(0, 1, min(B + 1, 2**11 + 1)), np.linspace(0, 1, I + 1) 


def compute_p(B, bc, f_source_str):
    # B, H1 and H2 must be related
    B = int((B / 4)) * 4
    grad = lambda u, h: (u[1:] - u[:-1]) / h

    # Choose base mesh 
    mesh_b = np.linspace(0, 1, B+1)
    hb = 1 / B
    u_hb = solver(mesh_b, bc, f_source_str)
    grad_hb = grad(u_hb, hb)

    factor = 2
    H1 = B // factor
    H2 = B // (factor * factor)

    mesh_1 = np.linspace(0, 1, H1 + 1)
    h1 = 1 / (H1)
    u_h1 = solver(mesh_1, bc, f_source_str)
    grad_h1 = grad(u_h1, h1)

    mesh_2 = np.linspace(0, 1, H2 + 1) 
    h2 = 1 / (H2)
    u_h2 = solver(mesh_2, bc, f_source_str)
    grad_h2 = grad(u_h2, h2)
   
    # note: mesh 1 is always coarser then mesh b. ratio of spacing is always 2 
    grad_h2_int = np.repeat(np.copy(grad_h2), factor)
    grad_h1_int = np.repeat(np.copy(grad_h1), factor)

    term1 = np.sqrt(np.sum((grad_h2_int- grad_h1)**2*h1))
    # print("TERM1", term1)
    term2 = np.sqrt(np.sum((grad_h1_int - grad_hb)**2*hb))
    # print("TERM2", term2)
    p = np.log(term1/term2)/np.log((factor))

    return abs(p)


def f_expr(a_k, freq, x):
    f = 0
    for i in range(len(a_k)):
        f += a_k[i] * np.sin(freq[i] * x)
    return f


def get_jump(u, step):
    grad_temp = (u[1:] - u[:-1]) / step
    grad = {"i-1": grad_temp[0:-2], "i": grad_temp[1:-1], "i+1": grad_temp[2:]}
    grad_jump_left = grad["i-1"] - grad["i"]
    grad_jump_right = grad["i"] - grad["i+1"]

    norm = np.linalg.norm(np.vstack((grad_jump_left, grad_jump_right)).transpose(), axis=1) 
    return norm


def element_residual(step, a_k, freq, mesh):
    f = partial(f_expr, a_k, freq)
    residual = lambda z: f(step * z + mesh[:-1]) * f(step * z + mesh[:-1]) * step
    return integrate.quad_vec(residual, 0, 1)[0]  