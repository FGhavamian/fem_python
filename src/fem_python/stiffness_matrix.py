import numpy as np

from fem_python import config
from fem_python.shape_functions import get_shape_function
from fem_python.mesh import FEMMesh


def make_stiffness_matrix(fem_mesh: FEMMesh):
    stiffness_mat = np.zeros((fem_mesh.num_nodes, fem_mesh.num_nodes))

    # This process is called matrix assembly. We essentially add element stiffness matrices to
    # their appropriate location in the global stiffness matrix.
    for e in range(config.num_elements):

        # Note that attributes of each element can possibly be unique
        # for this reason, it is common to extract them while looping through elements
        element_length = fem_mesh.element_length
        element_elasticity_module = config.bar_elasticity_module
        element_area = config.bar_area

        element_nodes = fem_mesh.element_to_nodes_mapping[e]
        node_coords = [
            fem_mesh.node_coordinates[element_nodes[0]],
            fem_mesh.node_coordinates[element_nodes[1]],
        ]

        element_stiffness = element_elasticity_module * element_area / element_length

        ShapeFunction = get_shape_function()
        shape_function = ShapeFunction(node_coords)
        b = shape_function.evaluate_b_at(0)
        jacob = shape_function.evaluate_jacob_at(0)

        # Gaussian integration has two components, location and weight.
        # For a 1D element, since the derivateive of shape function b is constant, the location is irrelevant
        gaussian_weight = 2

        element_stiffness_mat = (
            element_stiffness * np.dot(b.T, b) * jacob * gaussian_weight
        )

        stiffness_mat[e : e + 2, e : e + 2] += element_stiffness_mat

    return stiffness_mat


if __name__ == "__main__":
    # quick test
    print(make_stiffness_matrix(FEMMesh()))
