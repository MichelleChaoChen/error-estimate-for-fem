import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def plot_refinement(x, exact, meshes, solutions, est_global_errors, ex_global_errors, N_elements, color=True):
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
    plt.savefig(f"plots/refinement_plot")        
