import numpy as np

from fem_python.mesh import FEMMesh


def apply_dirichlet_boundary_condition(
    fem_mesh: FEMMesh, stiffness_mat, force_vec, displacement_step_x
):
    elements = fem_mesh.boundary_connectivity_matrices["right"]

    # The procedure of applying prescribed displacement is as follows:
    # Let's say prescribed displacements are applied at dofs_p.
    # Then, the system of equations can be written as following blocks:
    # [[k_bb , kbp], [k_pb , kpp]] [u_b, u_p] = [f_b, f_p]
    # if u_p is set to U, then system of equation can be written as:
    # [[k_bb , 0], [k_pb , 1]] [u_b, u_p] = [f_b, U]
    # Note that by solving the above equation, u_p becomes U.
    # The benefit of this formulation is that we do not change the dimnesion of the stiffness matrix.
    dofs_ux = []
    for nodes in elements:
        for node in nodes:
            dof_x = 2 * node
            dofs_ux.append(dof_x)

    stiffness_mat[np.ix_(dofs_ux)] = 0
    stiffness_mat[np.ix_(dofs_ux), np.ix_(dofs_ux)] = 1

    force_vec[np.ix_(dofs_ux)] = displacement_step_x

    # Here we constrain the nodes on the left side of the bar in the x direction.
    # But we avoid constraining all of the nodes in the y direction. This is to simulate the behavior of a 1D bar.
    # In a 1D bar, the nodes are free to move in the y direction.
    elements = fem_mesh.boundary_connectivity_matrices["left"]

    for nodes in elements:
        node_coords = []
        for node in nodes:
            node_coord = fem_mesh.node_coords[node]
            node_coords.append(node_coord)
        node_coords = np.array(node_coords)

        for node in nodes:
            # Note each node has two degrees of freedom. dof_x, dof_y.
            dof_x = 2 * node

            stiffness_mat[dof_x, :] = 0
            stiffness_mat[:, dof_x] = 0
            stiffness_mat[dof_x, dof_x] = 1

            force_vec[dof_x] = 0

    # To simulate the behavior of a 1D bar, we constaint only one node in the y direction.
    node = elements[0]
    dof_y = 2 * node + 1

    stiffness_mat[dof_y, :] = 0
    stiffness_mat[:, dof_y] = 0
    stiffness_mat[dof_y, dof_y] = 1

    force_vec[dof_y] = 0

    return stiffness_mat, force_vec


if __name__ == "__main__":
    # quick test
    force_vec = apply_neuman_boundary_condition(FEMMesh())
    print(force_vec)
