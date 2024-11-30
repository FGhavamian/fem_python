import numpy as np


def make_stiffness_matrix():
    element_elasticity_module = 1
    element_area = 1

    # Note that since we have two elements, the lengths of the bar which is 1 is divided by two.
    # In the previous example, we only had one element. So, the length of the bar was 1.
    element_length = 0.5

    element_stiffness = element_elasticity_module * element_area / element_length

    element1_stiffness_mat = element_stiffness * np.array([[1, -1], [-1, 1]])
    element2_stiffness_mat = element_stiffness * np.array([[1, -1], [-1, 1]])

    stiffness_mat = np.zeros((3, 3))

    # This process is called matrix assembly. We essentially add element stiffness matrices to
    # their appropriate location in the global stiffness matrix.
    stiffness_mat[:2, :2] = element1_stiffness_mat
    stiffness_mat[1:, 1:] += element2_stiffness_mat

    return stiffness_mat


if __name__ == "__main__":
    # quick test
    print(make_stiffness_matrix())
