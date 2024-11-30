import numpy as np

# make force vector (apply neuman boundary condition)
force_vec = np.array([0, 1])

# make stiffness vector
stiffness_mat = np.array([[1, -1], [-1, 1]])

# apply dirichlet boundary condition
stiffness_mat[0, 1:] = 0
stiffness_mat[1:, 0] = 0
stiffness_mat[0, 0] = 1

force_vec[0] = 0

# solve for displacement vector
u = np.linalg.solve(stiffness_mat, force_vec)

# output solution
print(u)
