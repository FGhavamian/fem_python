from fem_python.fem.stiffness_matrix import make_stiffness_matrix
from fem_python.fem.boundary_conditions import (
    apply_neuman_boundary_condition,
    apply_dirichlet_boundary_condition,
)
from fem_python.mesh.mesh import FEMMesh
from fem_python.fem.solver import solve
from fem_python.postprocess.postprocess import write_to_vtk

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

write_to_vtk(displacement_vec, force_vec, fem_mesh)
