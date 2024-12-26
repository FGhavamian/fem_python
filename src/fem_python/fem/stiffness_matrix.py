from typing import List

import numpy as np

from fem_python.fem.shape_functions import get_shape_function
from fem_python.mesh.mesh import FEMMesh
from fem_python.fem.integration import get_gauss_integration_setting
from fem_python.fem.material_model import AbstractMaterialModel
from fem_python import config


def make_stiffness_matrix_and_internal_force_vector(
    fem_mesh: FEMMesh, displacement_vec, materials: List[List[AbstractMaterialModel]]
):
    # Integration points are used to integrate stiffness matrices.
    # Stiffness matrix consists of polynomial shap functions. Ideally,
    # we would like to have a number of integration points that solve for the
    # integral exactly.
    integration_points = get_gauss_integration_setting(
        num_int_points=config.num_integration_points
    )

    # The problem we are considering has two degrees of freedom (ux, uy).
    # It essentially means that we allow each node to move in the x and y direction.
    # Note than an FEM model can have many more degrees of freedom, based on the physics of the problem
    num_dofs = fem_mesh.num_nodes * 2
    stiffness_mat = np.zeros((num_dofs, num_dofs))
    internal_force_vec = np.zeros((num_dofs,))

    for e in range(fem_mesh.num_elements):
        for i, integration_point in enumerate(integration_points):
            # The FEM mesh is defined by a tables of node coordinates and a connectivity matrix.
            # Here we use the connectivity matrix to the get nodes associated with an element.
            # We then extract coordinates of each node from the node coordinates table
            nodes = fem_mesh.connectivity_matrix[e]
            node_coords = []
            for node in nodes:
                node_coord = fem_mesh.node_coords[node]
                node_coords.append(node_coord)
            node_coords = np.array(node_coords)

            shape_function = get_shape_function(node_coords, config.element_type)
            b = shape_function.evaluate_b_at(integration_point.point)
            jacob_det = shape_function.evaluate_jacob_determinant_at(
                integration_point.point
            )

            material = materials[e][i]

            dofs = []
            for node in nodes:
                dofs += [2 * node, 2 * node + 1]

            element_displacement = displacement_vec[np.ix_(dofs)]
            element_strain = np.dot(b, element_displacement)

            material.update(element_strain)

            element_stiffness = materials[e][i].stiffness_matrix
            element_stress = materials[e][i].stress

            element_stiffness_mat = (
                np.dot(np.dot(b.T, element_stiffness), b)
                * jacob_det
                * integration_point.weight
            )

            element_internal_force = (
                np.dot(b.T, element_stress) * jacob_det * integration_point.weight
            )

            stiffness_mat[np.ix_(dofs, dofs)] += element_stiffness_mat
            internal_force_vec[np.ix_(dofs)] += element_internal_force

    return stiffness_mat, internal_force_vec


if __name__ == "__main__":
    pass
