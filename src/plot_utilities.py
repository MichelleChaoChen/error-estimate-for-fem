import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from utils import process_training_data
import pandas as pd

def plot_refinement(x, exact, meshes, solutions, est_global_errors, ex_global_errors, N_elements, plot_title, color=True):
    """
    Plots how the mesh is refined and shows the global error estimates
    over iterations of refinement before the global error falls below
    the desired threshold. 

    :param x: The domain 
    :param exact: The exact solution 
    :param meshes: Meshes of each iteration of refinement
    :param solutions: The FEM solution at each iteration
    :param est_global_errors: The estimated global error at each iteration
    :param ex_global_errors: The exact global error at each iteration
    :param N_elements: The size of the mesh
    :param color: Whether a color scheme should be used plotting, defaults to True
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


def compute_mse_error_estimate(data):
    """
    Computes the MSE in error estimation, 
    i.e. how far away is the estimated error
    from the exact error.

    :param data: Data from multiple AMR runs
    :return: MSE of error estimation over numerous iterations
    """
    mses = []
    for it, values in data.items(): 
        square_dif = list(map(lambda value: (value[1] - value[2]) * (value[1] - value[2]), values))
        mse = np.sqrt(np.sum(square_dif) / len(values))
        mses.append(mse)
    return mses 


def compute_average_elements(data):
    """
    Computes the average mesh size
    over multiple AMR runs for different 
    source functions. 

    :param data: Data from multiple AMR runs
    :return: THe average mesh size over the runs
    """   
    num_elements = []
    for it, values in data.items(): 
        values = np.array(values)
        n_elements = values[:, 0]
        mean_elements = np.mean(n_elements) 
        num_elements.append(mean_elements)
    return np.array(num_elements)


def plot_mse_error_estimate(data_nn, data_rec):
    """
    Plots the error (MSE) of the neural network error estimator 
    and the recovery-based error estimator. Provides a visual comparison of the two methods. 
    The results can be found in the experiments directory. 

    :param data_nn: AMR data with neural network error estimator
    :param data_rec: AMR data with recovery-based error estimator
    """
    mse_nn = compute_mse_error_estimate(data_nn)
    mse_rec = compute_mse_error_estimate(data_rec)
    plt.figure()
    plt.semilogy(range(1, len(mse_nn) + 1), mse_nn, 'o-', label='Neural Network')
    plt.semilogy(range(1, len(mse_rec) + 1), mse_rec, 'o-', label='Residual Method')
    plt.xticks(range(1, len(mse_nn) + 1))
    plt.xlabel('Iteration')
    plt.ylabel('Error in Error Estimation')
    plt.legend()
    plt.savefig('experiments/mse.svg')


def plot_number_elements(data_nn, data_rec):
    """
    Plots how the average mesh size changes 
    as the number of iterations in AMR increases. This 
    offers a comparison between mesh size with the neural network
    and recovery method. 

    :param data_nn: AMR data with neural network error estimator
    :param data_rec: AMR data with recovery-based error estimator
    """
    elements_nn =  compute_average_elements(data_nn)
    elements_rec = compute_average_elements(data_rec)
    plt.figure()
    max_size = max(elements_nn.shape[0], elements_rec.shape[0])
    elements_rec = np.pad(elements_rec, (0, max_size - elements_rec.shape[0]), mode='maximum')
    elements_nn = np.pad(elements_nn, (0, max_size - elements_nn.shape[0]), mode='maximum')
    relative_change = ((elements_rec / elements_nn) * 100.0) - 100.0
    plt.plot(range(1, len(elements_nn) + 1), relative_change , 'o--', label='Saved by Neural Network')
    plt.xticks(range(1, len(elements_nn[:-1]) + 1))
    plt.xlabel('Iteration')
    plt.ylabel('Percent of Mesh Size Reduction')
    plt.legend()
    plt.savefig('experiments/num_elements.svg')


def plot_average_iterations(avg_run_nn, avg_run_rec):
    """
    Plots the average number of iterations needed
    by neural network estimator and recovery-based
    estimator in AMR for the global error to fall
    below the desired threshold. 

    :param avg_run_nn: Average number of runs with neural network estimator
    :param avg_run_rec: Average number of runs with recovery-based estimator
    """
    methods = ['Neural Network', 'Residual Method']
    avg_run = [avg_run_nn, avg_run_rec]
    plt.figure()
    plt.bar(methods, height=avg_run, width=0.25, label='Average Number of Iterations')
    plt.legend()
    plt.savefig('experiments/avg_iterations.svg')


def plot_tuning_comparison(tuned, non_tuned, tp):
    tuned_loss, tuned_val = process_training_data(tuned)
    loss, val = process_training_data(non_tuned)
    
    losses = np.array([np.array(tuned_loss), np.array(val)])
    df = pd.DataFrame({"Tuned": losses[0], "Non_tuned":losses[1]})

    plt.figure()
    df.Tuned.rolling(5).mean().plot(logy=True)
    df.Non_tuned.rolling(5).mean().plot(logy=True)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'experiments/{tp}_tune_comparison.svg')


if __name__ == '__main__':
    plot_tuning_comparison('experiments/training-coarse-tuned.txt', 'experiments/training-coarse.txt', 'coarse')
    plot_tuning_comparison('experiments/training-fine-tuned.txt', 'experiments/training-fine.txt', 'fine')
