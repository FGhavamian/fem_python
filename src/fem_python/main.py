import numpy as np

from fem_python.fem import (
    make_stiffness_matrix,
    solve,
    apply_dirichlet_boundary_condition,
    apply_neuman_boundary_condition,
)
from fem_python.mesh.mesh import FEMMesh
from fem_python.postprocess.postprocess import (
    write_to_vtk,
    compute_stress_and_strain_at_nodes,
    compute_displacement_at_nodes,
)
from fem_python import config

# we run this once at the begining of the FEM code
# then we use mesh information during the runtime
fem_mesh = FEMMesh()

# We apply the external force or displacement at uniform proportions.
# For instance, if the presribed displacement is 1 and the number of time steps is 10,
# then the displacement at each time step will be 0.1.

force_step = np.array(config.uniform_force_at_right_boundary) / config.num_time_steps
displacement_step_x = (
    config.prescribed_displacement_at_right_boundary_x / config.num_time_steps
)

for t in range(config.num_time_steps):

    # make stiffness vector
    stiffness_mat = make_stiffness_matrix(fem_mesh)

    # apply neuman boundary condition
    force_vec = apply_neuman_boundary_condition(fem_mesh, force_step)

    # apply dirichlet boundary condition
    stiffness_mat, force_vec = apply_dirichlet_boundary_condition(
        fem_mesh, stiffness_mat, force_vec, displacement_step_x
    )

    # solve for displacement vector
    displacement_vec = solve(stiffness_mat, force_vec)


stress_vec, strain_vec = compute_stress_and_strain_at_nodes(displacement_vec, fem_mesh)
displacement_vec = compute_displacement_at_nodes(displacement_vec, fem_mesh)

vecs_dict = {
    "stress": stress_vec,
    "strain": strain_vec,
    "displacement": displacement_vec,
}

write_to_vtk(vecs_dict, fem_mesh)
