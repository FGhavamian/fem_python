import numpy as np


def solve(stiffness_mat, force_vec):
    displacement_vec = np.linalg.solve(stiffness_mat, force_vec)
    return displacement_vec
