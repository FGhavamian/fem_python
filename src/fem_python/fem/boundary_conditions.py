import numpy as np

from fem_python.mesh.mesh import FEMMesh
from fem_python.fem.shape_functions import get_shape_function
from fem_python.fem.integration import get_gauss_integration_setting
from fem_python import config


def apply_neuman_boundary_condition(fem_mesh: FEMMesh):
    num_dofs = fem_mesh.num_nodes * 2
    force_vec = np.zeros((num_dofs,))

    # Building force vector is much like building the stiffness matrix.
    # The difference is that instead of looping through elements, we loop through
    # boundary elements. And for each boundary element, we compute the force vector.
    # And just like the stiffness matrix assembly, we assemble the force vector.
    # Note that the elements on the right side of the bar are subjected to the force.
    elements = fem_mesh.boundary_connectivity_matrices["right"]

    integration_points = get_gauss_integration_setting(num_int_points=1)

    for nodes in elements:
        for integration_point in integration_points:
            node_coords = []
            for node in nodes:
                node_coord = fem_mesh.node_coords[node]
                node_coords.append(node_coord)
            node_coords = np.array(node_coords)

        shape_function = get_shape_function(node_coords, element_type="L2")
        n = shape_function.evaluate_n_at(integration_point.point)
        jacob_det = shape_function.evaluate_jacob_determinant_at(
            integration_point.point
        )

        unform_force_vec = np.array(config.uniform_force_at_right_boundary)

        element_force_vec = (
            np.dot(n.T, unform_force_vec) * jacob_det * integration_point.weight
        )

        # Note that the ouput of the dot product above is a 2D array. The
        # second dimension has the size of 1. This is because, element_force_vec is
        # actually a vector and not a matrix. We turn the 2D array into a 1D array by
        # squeezing it.
        element_force_vec = np.squeeze(element_force_vec)

        dofs = []
        for node in nodes:
            dofs += [2 * node, 2 * node + 1]

        force_vec[np.ix_(dofs)] += element_force_vec

    return force_vec


def apply_dirichlet_boundary_condition(fem_mesh: FEMMesh, stiffness_mat, force_vec):
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

            stiffness_mat[dof_x, 1:] = 0
            stiffness_mat[1:, dof_x] = 0
            stiffness_mat[dof_x, dof_x] = 1

            force_vec[dof_x] = 0

    # To simulate the behavior of a 1D bar, we constaint only one node in the y direction.
    node = elements[0]
    dof_y = 2 * node + 1

    stiffness_mat[dof_y, 1:] = 0
    stiffness_mat[1:, dof_y] = 0
    stiffness_mat[dof_y, dof_y] = 1

    force_vec[dof_y] = 0

    return stiffness_mat, force_vec


if __name__ == "__main__":
    # quick test
    force_vec = apply_neuman_boundary_condition(FEMMesh())
    print(force_vec)
