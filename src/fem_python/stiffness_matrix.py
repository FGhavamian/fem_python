import numpy as np

from fem_python import config


def make_stiffness_matrix():
    element_elasticity_module = config.bar_elasticity_module
    element_area = config.bar_area

    # two elements, then length of each is 1/2
    # three elements, then length of each is 1/3
    # and so on ...
    element_length = config.bar_length / config.num_elements

    # in a 1D linear bar, the number of nodes is one more than the number of elements
    num_nodes = config.num_elements + 1

    stiffness_mat = np.zeros((num_nodes, num_nodes))

    # This process is called matrix assembly. We essentially add element stiffness matrices to
    # their appropriate location in the global stiffness matrix.
    for e in range(config.num_elements):
        element_stiffness = element_elasticity_module * element_area / element_length

        element_stiffness_mat = element_stiffness * np.array([[1, -1], [-1, 1]])

        # The location of the first element is at nodes 0:1, 0:1 in the global stiffness matrix
        # second element: 1:2, 1:2
        # third element: 2:3, 2:3
        # Note the overlap. The node at index 1 is common between first element and second element
        stiffness_mat[e : e + 2, e : e + 2] += element_stiffness_mat

    return stiffness_mat


if __name__ == "__main__":
    # quick test
    print(make_stiffness_matrix())
