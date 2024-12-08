import numpy as np

from fem_python.mesh import FEMMesh


def apply_neuman_boundary_condition(fem_mesh: FEMMesh):
    force_vec = np.zeros((fem_mesh.num_dofs,))

    # Note that the external force is applied to the end of the bar
    # The end of the bar coincides with the last node
    nodes = fem_mesh.boundary_nodes_on_the_right
    for node in nodes:
        dof_x = fem_mesh.node_to_dof_mapping[node][0]
        force_vec[dof_x] = 1 / 2

    return force_vec


def apply_dirichlet_boundary_condition(fem_mesh: FEMMesh, stiffness_mat, force_vec):
    # Here we constrain the nodes on the left side of the bar in the x direction.
    # But we avoid constraining all of the nodes in the y direction. This is to simulate the behavior of a 1D bar.
    # In a 1D bar, the nodes are free to move in the y direction.
    nodes = fem_mesh.boundary_nodes_on_the_left
    for node in nodes:
        dof_x = fem_mesh.node_to_dof_mapping[node][0]

        stiffness_mat[dof_x, 1:] = 0
        stiffness_mat[1:, dof_x] = 0
        stiffness_mat[dof_x, dof_x] = 1

        force_vec[dof_x] = 0

    # To simulate the behavior of a 1D bar, we constaint only one node in the y direction.
    node = nodes[0]
    dof_y = fem_mesh.node_to_dof_mapping[node][1]

    stiffness_mat[dof_y, 1:] = 0
    stiffness_mat[1:, dof_y] = 0
    stiffness_mat[dof_y, dof_y] = 1

    force_vec[dof_y] = 0

    return stiffness_mat, force_vec


if __name__ == "__main__":
    # quick test
    force_vec = apply_neuman_boundary_condition(FEMMesh())
    print(force_vec)
