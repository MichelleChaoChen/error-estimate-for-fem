
from numpy import sqrt, arange
EPSILON = 1e-4


#TODO: test througout with real data 
def approx(x, y):
    return abs(x - y) <= EPSILON

def refine(mesh, err_pred, global_error):
    # base case
    if len(mesh) == 1:
        return mesh

    num_elements = len(mesh)
    compute_err = lambda err : err * sqrt(num_elements) / global_error

    refined_mesh = [mesh[0]]
    for i in range(0, num_elements - 1):
        curErr = compute_err(err_pred[i])
        num_points = int(round(curErr))

        refined_mesh.extend(
            mesh[i] + (mesh[i+1] - mesh[i]) / (num_points + 1) * arange(1, num_points + 2)
        )

        if approx(curErr, 0.5) and \
                (i + 1 < len(err_pred) and approx(compute_err(err_pred[i+1]), 0.5)):
            refined_mesh.pop()

    return refined_mesh

if __name__ == '__main__':
    refined_mesh = refine([0,0.5,1], [0.7, 0.2], 0.5)
    print(refined_mesh)