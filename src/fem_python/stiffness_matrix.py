import numpy as np

from fem_python import config
from fem_python.shape_functions import get_shape_function
from fem_python.mesh import FEMMesh
from fem_python.integration import get_gauss_integration_setting
from fem_python.material_model import get_elastic_stiffness_matrix


def make_stiffness_matrix(fem_mesh: FEMMesh):
    # Integration points are used to integrate stiffness matrices.
    # Stiffness matrix consists of polynomial shap functions. Ideally,
    # we would like to have a number of integration points that solve for the
    # integral exactly.
    integration_points = get_gauss_integration_setting(
        num_int_points=fem_mesh.num_integration_points_needed
    )

    stiffness_mat = np.zeros((fem_mesh.num_dofs, fem_mesh.num_dofs))

    for e in range(config.num_elements):
        for integration_point in integration_points:
            element_stiffness = get_elastic_stiffness_matrix()

            node_coords = fem_mesh.get_node_coords_for_element(e)
            shape_function = get_shape_function(node_coords)

            b = shape_function.evaluate_b_at(integration_point.point)
            jacob_det = shape_function.evaluate_jacob_determinant_at(
                integration_point.point
            )

            element_stiffness_mat = (
                np.dot(np.dot(b.T, element_stiffness), b)
                * jacob_det
                * integration_point.weight
            )

            # This process is called matrix assembly. We essentially add element stiffness matrices to
            # their appropriate location in the global stiffness matrix.
            element_nodes = fem_mesh.element_to_nodes_mapping[e]
            dofs = []
            for node in element_nodes:
                dofs += fem_mesh.node_to_dof_mapping[node]

            stiffness_mat[np.ix_(dofs, dofs)] += element_stiffness_mat

    return stiffness_mat


if __name__ == "__main__":
    # quick test
    print(make_stiffness_matrix(FEMMesh()).shape)
    # make_stiffness_matrix(FEMMesh())
