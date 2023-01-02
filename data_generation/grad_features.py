import numpy as np

'''
Start with base grid 40 elements
x = [0, x1, x2 ... xn]
grad = [g1, g2, g3 ... gn]


New grid let with Y elements, Y > 40
x= [0, X1, X2, ... XN]
grad = [G1, G2, .... GN]


Calculate error based on energy method
'''
# x_20 = np.loadtxt("data_x_20.csv", delimiter=',')
# x_100 = np.loadtxt("data_x_100.csv", delimiter=',')

# grad_20 = np.loadtxt("data_grad_20.csv", delimiter=',')
# grad_100 = np.loadtxt("data_grad_100.csv", delimiter=',')
def factor(grad_new, grad_old):
    return np.abs(((grad_new - grad_old) / grad_old) * 100)

def gendata(new_sol, new_grid, old_sol, old_grid, error, sampling_freq):



    old_grad = (old_sol[1:] - old_sol[:-1])/(old_grid[1:] - old_grid[:-1]) 
    new_grad = (new_sol[1:] - new_sol[:-1])/(new_grid[1:] - new_grid[:-1]) 

    new_step = new_grid[1:] - new_grid[:-1]
    old_step = old_grid[1:] - old_grid[:-1]


    index_grad = np.searchsorted(old_grid, new_grid, 'left')[1:]
    inter_old_grad = np.array(list((map(lambda x: old_grad[x-1], index_grad))))
    inter_old_grid = np.array(list(map(lambda x: old_step[x-1], index_grad)))
    factor_grad = factor(new_grad, inter_old_grad)
    factor_step = factor(new_step, inter_old_grid)



    factor_step_im1 = factor_step[0:-2:sampling_freq]
    factor_step_i = factor_step[1:-1:sampling_freq]
    factor_step_ip1 = factor_step[2::sampling_freq]

    factor_grad_im1 = np.log(factor_grad[0:-2:sampling_freq])
    factor_grad_i = np.log(factor_grad[1:-1:sampling_freq])
    factor_grad_ip1 = np.log(factor_grad[2::sampling_freq])



    grad_im1 = new_grad[0:-2:sampling_freq]
    grad_i = new_grad[1:-1:sampling_freq]
    grad_ip1 =new_grad[2::sampling_freq]
    jump_1 = np.log(np.abs(grad_im1 - grad_i))
    jump_2 = np.log(np.abs(grad_i - grad_ip1)) 
    
    error_sam = np.log(error[1:-1:sampling_freq])

    data = np.vstack((factor_step_im1, factor_step_i, factor_step_ip1, \
                    factor_grad_im1, factor_grad_i, factor_grad_ip1, \
                    jump_1, jump_2, error_sam))

    return data



