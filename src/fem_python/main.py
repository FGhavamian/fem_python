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
stiffness_mat, force_vec = apply_dirichlet_boundary_condition(
    fem_mesh, stiffness_mat, force_vec
)

# solve for displacement vector
displacement_vec = solve(stiffness_mat, force_vec)

# output solution
import pandas as pd

results = {"x": [], "y": [], "ux": [], "uy": [], "fx": [], "fy": []}

for node in range(fem_mesh.num_nodes):
    dof_x = 2 * node
    dof_y = 2 * node + 1

    x, y = fem_mesh.node_coords[node]

    results["x"].append(x)
    results["y"].append(y)
    results["ux"].append(displacement_vec[dof_x])
    results["uy"].append(displacement_vec[dof_y])
    results["fx"].append(force_vec[dof_x])
    results["fy"].append(force_vec[dof_y])


print(pd.DataFrame(results).sort_values(by="y"))
