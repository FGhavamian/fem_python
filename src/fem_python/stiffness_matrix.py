import numpy as np

from fem_python import config
from fem_python.shape_functions import get_shape_function
from fem_python.mesh import FEMMesh
from fem_python.integration import get_gauus_integration_setting


def make_stiffness_matrix(fem_mesh: FEMMesh):
    # Integration points are used to integrate stiffness matrices.
    # Stiffness matrix consists of polynomial shap functions. Ideally,
    # we would like to have a number of integration points that solve for the
    # integral exactly.
    integration_points = get_gauus_integration_setting(
        num_int_points=fem_mesh.num_integration_points_needed
    )

    stiffness_mat = np.zeros((fem_mesh.num_nodes, fem_mesh.num_nodes))

    for e in range(config.num_elements):
        for integration_point in integration_points:
            # Note that attributes of each element can possibly be unique
            # for this reason, it is common to extract them while looping through elements
            element_elasticity_module = config.bar_elasticity_module
            element_area = config.bar_area

            node_coords = fem_mesh.get_node_coords_for_element(e)

            element_stiffness = element_elasticity_module * element_area

            ShapeFunction = get_shape_function()
            shape_function = ShapeFunction(node_coords)

            b = shape_function.evaluate_b_at(integration_point.location)
            jacob = shape_function.evaluate_jacob_at(integration_point.location)

            element_stiffness_mat = (
                element_stiffness * np.dot(b.T, b) * jacob * integration_point.weight
            )

            # This process is called matrix assembly. We essentially add element stiffness matrices to
            # their appropriate location in the global stiffness matrix.
            element_nodes = fem_mesh.element_to_nodes_mapping[e]

            stiffness_mat[np.ix_(element_nodes, element_nodes)] += element_stiffness_mat

    return stiffness_mat


if __name__ == "__main__":
    # quick test
    print(make_stiffness_matrix(FEMMesh()))
    # make_stiffness_matrix(FEMMesh())
