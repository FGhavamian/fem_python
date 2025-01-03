import numpy as np
import matplotlib.pyplot as plt

from fem_python.fem import (
    make_stiffness_matrix_and_internal_force_vector,
    solve,
    apply_dirichlet_boundary_condition,
    get_material_model,
)
from fem_python.fem.integration import get_gauss_integration_setting
from fem_python.mesh.mesh import FEMMesh
from fem_python.postprocess import (
    write_to_vtk,
    compute_stress_and_strain_at_nodes,
    compute_displacement_at_nodes,
)

from fem_python import config

# we run this once at the begining of the FEM code
# then we use mesh information during the runtime
fem_mesh = FEMMesh()

# Note each element, in principle, is allowed to have a different material model.
# We compute stress and stiffeness matrix for each integration point in each element.
# That is why we initiate a material model for integration point.
materials = []
for _ in range(fem_mesh.num_elements):
    material_integration_point = []

    num_integration_points = len(
        get_gauss_integration_setting(config.num_integration_points)
    )
    for _ in range(num_integration_points):
        material_integration_point.append(
            get_material_model(
                config.material_model_name,
                elasticity_module=config.bar_elasticity_module,
                poission_ratio=config.bar_poission_ratio,
            )
        )

    materials.append(material_integration_point)

# We apply the prescribed displacement at uniform proportions.
# For instance, if the presribed displacement is 1 and the number of time steps is 10,
# then the displacement at each time step will be 0.1.
displacement_step_x_increment = (
    config.prescribed_displacement_at_right_boundary_x / config.num_time_steps
)

# We initiate the displacement vector here because, the function make_stiffness_matrix_and_internal_force_vector
# requires the displacement vector to compute strain. A the first time step, the displacement vector is zero hence,
# the strain is zero. After that, the displacement vector is updated at each time step and hence, the strain takes a
# none-zero value.
displacement_vec = np.zeros((fem_mesh.num_nodes * 2,))

displacement_step_x = 0

# It is usually of interest to keep track of a quantity of interest. In the 1D bar problem,
# we are interested in the force and displacement at the right boundary. We keep track of
# how the internal force increases as we increase the displacement.
force_displacement_right_boundary = {"force": [], "displacement": []}

for t in range(config.num_time_steps):
    for i in range(config.max_num_nr_iterations):
        if i == 0:
            displacement_step_x += displacement_step_x_increment
        else:
            displacement_step_x = 0

        stiffness_mat, internal_force_vec = (
            make_stiffness_matrix_and_internal_force_vector(
                fem_mesh, displacement_vec, materials
            )
        )

        stiffness_mat, internal_force_vec = apply_dirichlet_boundary_condition(
            fem_mesh, stiffness_mat, internal_force_vec, displacement_step_x
        )

        displacement_vec_correction = solve(stiffness_mat, internal_force_vec)
        displacement_vec += displacement_vec_correction

        residual_norm = np.linalg.norm(internal_force_vec)
        print(f"Time step: {t}, Iteration: {i}, Residual: {residual_norm}")
        if residual_norm < 1e-6:
            break

    # The internal force that was computed in the last increment is nullified through application of
    # the boundary condition. The reason is that its values are zero at internal nodes and its values at
    # boundary nodes are set to zero by the boundary condition. We have to recompute it here.
    _, internal_force_vec = make_stiffness_matrix_and_internal_force_vector(
        fem_mesh, displacement_vec, materials
    )

    # Collecting the force and displacement at the right boundary. The force and the displacement vectors on the right boundary
    # are averaged out. This is because the right boundary is made up of multiple nodes. This is again not the most accurate way of
    # computing the force but, it is quite simple.
    elements = fem_mesh.boundary_connectivity_matrices["right"]

    dofs_ux = []
    for nodes in elements:
        for node in nodes:
            dof_x = 2 * node
            dofs_ux.append(dof_x)
    dofs_ux = list(set(dofs_ux))

    internal_force_right_boundary = np.sum(internal_force_vec[np.ix_(dofs_ux)])
    displacement_right_boundary = np.mean(displacement_vec[np.ix_(dofs_ux)])

    force_displacement_right_boundary["force"].append(internal_force_right_boundary)
    force_displacement_right_boundary["displacement"].append(
        displacement_right_boundary
    )

stress_vec, strain_vec = compute_stress_and_strain_at_nodes(fem_mesh, materials)
displacement_vec = compute_displacement_at_nodes(displacement_vec, fem_mesh)
internal_force_vec = compute_displacement_at_nodes(internal_force_vec, fem_mesh)

vecs_dict = {
    "stress": stress_vec,
    "strain": strain_vec,
    "displacement": displacement_vec,
    "internal_force": internal_force_vec,
}

write_to_vtk(vecs_dict, fem_mesh)

plt.plot(
    force_displacement_right_boundary["displacement"],
    force_displacement_right_boundary["force"],
    marker="o",
)
plt.xlabel("Displacement at right boundary")
plt.ylabel("Force at right boundary")
plt.title("1D Bar Problem")
plt.show()
