from typing import List

import numpy as np
import meshio

from fem_python.fem.shape_functions import get_shape_function
from fem_python.mesh import FEMMesh
from fem_python.fem.integration import get_gauss_integration_setting
from fem_python.fem.material_model import AbstractMaterialModel
from fem_python import config


def compute_stress_and_strain_at_nodes(
    fem_mesh: FEMMesh, materials: List[List[AbstractMaterialModel]]
):
    """We interpolate the stress and strain at the nodes. The reason is that the
    write_to_vtk function expects values at nodes. There should be more accurate ways to
    visualize stress/strain fields.

    For each integration point, we compute the stress and strain. We then average the values.
    """
    integration_points = get_gauss_integration_setting(
        num_int_points=config.num_integration_points
    )

    # Note that each node is shared among multiple elements. The values of stress and
    # strain can be different for each element. However, the true value of stress and strain at
    # a node is unique. To estimate this unique value, we collect all stress and strain values
    # contributed by neighboring elements and average them. This is not neccessary the most accurate way.
    # But it most certainly is the simplest way.
    node_stress = {n: [] for n in range(fem_mesh.num_nodes)}
    node_strain = {n: [] for n in range(fem_mesh.num_nodes)}

    for e in range(fem_mesh.num_elements):
        for i, _ in enumerate(integration_points):
            material = materials[e][i]

            nodes = fem_mesh.connectivity_matrix[e]

            for node in nodes:
                node_strain[node].append(material.strain)
                node_stress[node].append(material.stress)

    stress_vec = np.zeros((fem_mesh.num_nodes, 3))
    strain_vec = np.zeros((fem_mesh.num_nodes, 3))

    for node in range(fem_mesh.num_nodes):
        stress_vec[node] = np.mean(node_stress[node], axis=0)
        strain_vec[node] = np.mean(node_strain[node], axis=0)

    return stress_vec, strain_vec


def compute_displacement_at_nodes(displacement_vec, fem_mesh: FEMMesh):
    displacement = np.zeros((fem_mesh.num_nodes, 2))

    for node in range(fem_mesh.num_nodes):
        dof_x = 2 * node
        dof_y = 2 * node + 1

        displacement[node] = [displacement_vec[dof_x], displacement_vec[dof_y]]

    return displacement


def write_to_vtk(vecs_dict, fem_mesh: FEMMesh):
    meshio.write_points_cells(
        "outputs/plate_w_hole.vtk",
        fem_mesh.node_coords,
        fem_mesh.cells,
        point_data=vecs_dict,
    )
