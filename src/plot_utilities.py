import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def plot_refinement(x, exact, meshes, solutions, est_global_errors, ex_global_errors, N_elements, plot_title, color=True):
    """_summary_

    :param x: _description_
    :param exact: _description_
    :param meshes: _description_
    :param solutions: _description_
    :param est_global_errors: _description_
    :param ex_global_errors: _description_
    :param N_elements: _description_
    :param color: _description_, defaults to True
    """
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(211)
    plt.plot(x, exact, lw=6, zorder=0)
    plt.grid()
    
    iter = 0
    for i, mesh in meshes.items(): 
        ax2 = plt.subplot(212)
        
        plt.title(f"Iteration: {i}, Estimated Error: {est_global_errors[i - 1]:.4e}, Exact Error: {ex_global_errors[i - 1]:.4e}")
        
        if color:
            xy = np.vstack([mesh, solutions[i]])
            z = gaussian_kde(xy)(xy)
            ax2.scatter(mesh, i * np.ones(len(mesh)), c=z, cmap='rainbow')
        else:
            ax2.scatter(mesh, i * np.ones(len(mesh)))
        iter += 1
    
    if color:
        plt.colorbar(ax2.collections[0], location='bottom', shrink=0.7)
    
    plt.grid()
    ax1.scatter(meshes[iter], solutions[iter], s=5, color='r')
    plt.gca().set_yticks(np.arange(1, iter + 1))
    plt.gca().set_yticklabels([f"Iter: {i + 1}, Elements: {N_elements[i]}" for i in range(iter)])
    plt.tight_layout()
    plt.savefig(f"plots/refinement_plot_{plot_title}")        


def plot_error_convergence(ex_global_errors, est_global_errors_nn, est_global_errors_rec, plot_title):
    iterations = len(ex_global_errors)
    plt.figure()
    plt.semilogy(range(iterations), ex_global_errors, 'o-', color='orange', label='Exact Errors')
    plt.semilogy(range(iterations), est_global_errors_nn, 'o--', color='blue', label='Estimated Errors (NN)')
    plt.semilogy(range(iterations), est_global_errors_rec, 'o--', color='green', label='Estimated Errors (Recovery)')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Global Error')
    plt.title(plot_title)
    plt.savefig('experiments/error_plot.png')


def write_amr_data(filename, nr_elements, est_global_errors, ex_global_errors):
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
    f = open(filename, "r")
    rows = f.read().splitlines()[1:-1]
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


def compute_mse_error_estimate(data):    
    mses = []
    for it, values in data.items(): 
        square_dif = list(map(lambda value: (value[1] - value[2]) * (value[1] - value[2]), values))
        mse = np.sqrt(np.sum(square_dif) / len(values))
        mses.append(mse)
    return mses 


def compute_average_elements(data):    
    num_elements = []
    for it, values in data.items(): 
        values = np.array(values)
        n_elements = values[:, 0]
        mean_elements = np.mean(n_elements) 
        num_elements.append(mean_elements)
    return num_elements


def plot_mse_error_estimate(data_nn, data_rec):
    mse_nn = compute_mse_error_estimate(data_nn)
    mse_rec = compute_mse_error_estimate(data_rec)
    plt.figure()
    plt.semilogy(range(1, len(mse_nn) + 1), mse_nn, 'o-', label='Neural Network')
    plt.semilogy(range(1, len(mse_rec) + 1), mse_rec, 'o-', label='Recovery Method')
    plt.xticks(range(1, len(mse_nn) + 1))
    plt.xlabel('Iteration')
    plt.ylabel('Error in Error Estimation')
    plt.legend()
    plt.savefig('experiments/mse.svg')


def plot_number_elements(data_nn, data_rec):
    elements_nn = compute_average_elements(data_nn)
    elements_rec = compute_average_elements(data_rec)
    plt.figure()
    relative_change = (np.array(elements_rec) / np.array(elements_nn[:-1]) * 100.0) - 100.0
    plt.plot(range(1, len(elements_nn[:-1]) + 1), relative_change , 'o--', label='Saved by Neural Network')
    plt.xticks(range(1, len(elements_nn[:-1]) + 1))
    plt.xlabel('Iteration')
    plt.ylabel('Percent of Mesh Size Reduction')
    plt.legend()
    plt.savefig('experiments/num_elements.svg')


def plot_average_iterations(avg_run_nn, avg_run_rec):
    methods = ['Neural Network', 'Recovery Method']
    avg_run = [avg_run_nn, avg_run_rec]
    plt.figure()
    plt.bar(methods, height=avg_run, width=0.4, label='Average Number of Iterations')
    plt.legend()
    plt.savefig('experiments/avg_iterations.svg')