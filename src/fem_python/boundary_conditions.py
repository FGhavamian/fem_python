import numpy as np


def apply_neuman_boundary_condition():
    force_vec = np.array([0, 0, 1])
    return force_vec


def apply_dirichlet_boundary_condition(stiffness_mat, force_vec):
    stiffness_mat[0, 1:] = 0
    stiffness_mat[1:, 0] = 0
    stiffness_mat[0, 0] = 1

    force_vec[0] = 0

    return stiffness_mat, force_vec
