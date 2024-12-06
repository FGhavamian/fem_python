import numpy as np

from fem_python.stiffness_matrix import make_stiffness_matrix
from fem_python.boundary_conditions import (
    apply_neuman_boundary_condition,
    apply_dirichlet_boundary_condition,
)
from fem_python.mesh import FEMMesh
from fem_python.solver import solve

# we run this once at the begining of the FEM code
# then we use mesh information during the runtime
fem_mesh = FEMMesh()

# make stiffness vector
stiffness_mat = make_stiffness_matrix(fem_mesh)

# apply neuman boundary condition
force_vec = apply_neuman_boundary_condition(fem_mesh)

# apply dirichlet boundary condition
stiffness_mat, force_vec = apply_dirichlet_boundary_condition(stiffness_mat, force_vec)

# solve for displacement vector
displacement_vec = solve(stiffness_mat, force_vec)

# output solution
print(displacement_vec)
