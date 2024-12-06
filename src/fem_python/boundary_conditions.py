import numpy as np

from fem_python.mesh import FEMMesh


def apply_neuman_boundary_condition(fem_mesh: FEMMesh):
    force_vec = np.zeros((fem_mesh.num_nodes,))

    # Note that the external force is applied to the end of the bar
    # The end of the bar coincides with the last node
    force_vec[-1] = 1

    return force_vec


def apply_dirichlet_boundary_condition(stiffness_mat, force_vec):
    stiffness_mat[0, 1:] = 0
    stiffness_mat[1:, 0] = 0
    stiffness_mat[0, 0] = 1

    force_vec[0] = 0

    return stiffness_mat, force_vec


if __name__ == "__main__":
    # quick test
    force_vec = apply_neuman_boundary_condition()
    print(force_vec)
