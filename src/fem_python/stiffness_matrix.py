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

        # Note that the shape function has the shape of 1x2.
        # 1 is the dimentionality of the model (1D model)
        # and 2 is the number of nodes in the element. 1D linear element has 2 nodes.
        b = np.array([[1, -1]])

        # the jacobian allows us to transform values from a "naturial" coordinate to "physical" coordinate
        jacobian = element_length / 2

        # Gaussian integration has two components, location and weight.
        # For a 1D element, since the derivateive of shape function b is constant, the location is irrelevant
        gaussian_weight = 2

        element_stiffness_mat = (
            element_stiffness * np.dot(b.T, b) * jacobian * gaussian_weight
        )

        # The location of the first element is at nodes 0:1, 0:1 in the global stiffness matrix
        # second element: 1:2, 1:2
        # third element: 2:3, 2:3
        # Note the overlap. The node at index 1 is common between first element and second element
        stiffness_mat[e : e + 2, e : e + 2] += element_stiffness_mat

    return stiffness_mat


if __name__ == "__main__":
    # quick test
    print(make_stiffness_matrix())
