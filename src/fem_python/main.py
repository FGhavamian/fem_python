# from fem_python.fem.stiffness_matrix import make_stiffness_matrix
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

# we run this once at the begining of the FEM code
# then we use mesh information during the runtime
fem_mesh = FEMMesh()

# make stiffness vector
stiffness_mat = make_stiffness_matrix(fem_mesh)

# apply neuman boundary condition
force_vec = apply_neuman_boundary_condition(fem_mesh)

# apply dirichlet boundary condition
stiffness_mat, force_vec = apply_dirichlet_boundary_condition(
    fem_mesh, stiffness_mat, force_vec
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
