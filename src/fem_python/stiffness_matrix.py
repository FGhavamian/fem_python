import numpy as np

from fem_python import config
from fem_python.shape_functions import get_shape_function


def make_stiffness_matrix():
    element_elasticity_module = config.bar_elasticity_module
    element_area = config.bar_area

    # two elements, then length of each is 1/2
    # three elements, then length of each is 1/3
    # and so on ...
    element_length = config.bar_length / config.num_elements

    stiffness_mat = np.zeros((config.num_nodes, config.num_nodes))

    # This process is called matrix assembly. We essentially add element stiffness matrices to
    # their appropriate location in the global stiffness matrix.
    for e in range(config.num_elements):
        element_stiffness = element_elasticity_module * element_area / element_length

        element_nodes = np.array([e * element_length, (e + 1) * element_length])

        ShapeFunction = get_shape_function()
        shape_function = ShapeFunction(element_nodes)
        b = shape_function.evaluate_b_at(0)
        jacob = shape_function.evaluate_jacob_at(0)

        # Gaussian integration has two components, location and weight.
        # For a 1D element, since the derivateive of shape function b is constant, the location is irrelevant
        gaussian_weight = 2

        element_stiffness_mat = (
            element_stiffness * np.dot(b.T, b) * jacob * gaussian_weight
        )

        stiffness_mat[e : e + 2, e : e + 2] += element_stiffness_mat

    return stiffness_mat


if __name__ == "__main__":
    # quick test
    print(make_stiffness_matrix())
