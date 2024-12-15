from abc import ABC, abstractmethod

import numpy as np

from fem_python import config


class Point:
    def __init__(self, xi, eta):
        self.xi = xi
        self.eta = eta


class ShapeFunction(ABC):
    @abstractmethod
    def evaluate_n_at():
        pass

    @abstractmethod
    def evaluate_b_at():
        pass

    @abstractmethod
    def evaluate_jacob_determinant_at(self):
        """Note that in 2D setting the jacobian is a 2x2 matrix.
        Let's say:
            x = sum_i^n N_i x_i
            y = sum_i^n N_i y_i

        Then the differential of dx, dy becomes
            x = sum_i^n frac{partial N_i}{partial xi} dxi x_i + frac{partial N_i}{partial eta} deta x_i
            y = sum_i^n frac{partial N_i}{partial xi} dxi y_i + frac{partial N_i}{partial eta} deta y_i

        In the matrix from
            [dx, dy] = jacobian [dxi deta]

        Where jacobian is
            [[sum_i^n frac{partial N_i}{partial xi} x_i, frac{partial N_i}{partial xi} y_i],
             [sum_i^n frac{partial N_i}{partial eta} x_i, frac{partial N_i}{partial eta} y_i]]
        """
        pass


class L2ShapFunction(ShapeFunction):
    def __init__(self, nodes):
        self.nodes = nodes

    def evaluate_n_at(self, int_point: Point):
        n_mat, _ = self._get_shape_functions_and_their_derivatives(int_point)

        n_mat = np.array(
            [
                [n_mat[0], 0, n_mat[1], 0],
                [0, n_mat[0], 0, n_mat[1]],
            ]
        )

        return n_mat

    def evaluate_b_at(self, int_point: Point):
        raise NotImplementedError(
            "B matrix is 2D for a line element is not implemented"
        )

    def evaluate_jacob_determinant_at(self, int_point: Point):
        jacobian = self._evaluate_jacob_at(int_point)

        # When the linear element is defined in the 2D space, the jacobian becomes rectangular.
        # We cannot compute jacobian for a rectangular matrix. Instead we consider what the determinant of
        # Jacobian was used for. It was essentially a measure of incremental difference between the element volume
        # (in this case length), in physical space and that of the isoparametric space.
        return np.sqrt(jacobian[0] ** 2 + jacobian[1] ** 2)

    def _evaluate_jacob_at(self, int_point: Point):
        _, dn_mat = self._get_shape_functions_and_their_derivatives(int_point)
        jacobian = dn_mat.dot(self.nodes)
        return jacobian

    def _get_shape_functions_and_their_derivatives(self, point: Point):
        # Note that the shape function matrix has two rows and four columns.
        # Each column is associated with a node. Each row is associated with x or y direction.
        # For instance, the value at (1,0) is the shape function of the second node associated with
        # the displacement (or force) in the x direction

        n1 = 0.5 * (1 - point.xi)
        n2 = 0.5 * (1 + point.xi)

        n_mat = np.array([n1, n2])

        dn1_dxi = -0.5
        dn2_dxi = 0.5

        dn_dxi_mat = np.array([dn1_dxi, dn2_dxi])

        return n_mat, dn_dxi_mat


class Q4ShapeFunction(ShapeFunction):
    def __init__(self, nodes):
        """
        Args:
            nodes (np.array): nodes should have the format of nx2. n is the number of nodes in the element and 2 is due to 2D model.
                              the nodes should be in the counter-clockwise order. Otherwise jacobian computation becomes faulty.
        """
        self.nodes = nodes

    def evaluate_n_at(self, int_point: Point):
        n_mat, _ = self._get_shape_functions_and_their_derivatives(int_point)

        n_mat = np.array(
            [
                [n_mat[0], 0, n_mat[1], 0, n_mat[2], 0, n_mat[3], 0],
                [0, n_mat[0], 0, n_mat[1], 0, n_mat[2], 0, n_mat[3]],
            ]
        )

        return n_mat

    def evaluate_b_at(self, int_point: Point):
        jacobian = self._evaluate_jacob_at(int_point)
        inv_jacob = np.linalg.inv(jacobian)

        _, dn_dxi_mat = self._get_shape_functions_and_their_derivatives(int_point)

        dn_dx_mat = inv_jacob.dot(dn_dxi_mat)

        # The B matrix is a 3x8 matrix.
        # The reason is that B matrix is designed to be multiplied with the displacement vector of nodes
        # and produce the strain vector [epsilon_xx, epsilon_yy, epsilon_xy]. That's why it has 3 rows.
        dn1_dx = dn_dx_mat[0, 0]
        dn1_dy = dn_dx_mat[1, 0]
        dn2_dx = dn_dx_mat[0, 1]
        dn2_dy = dn_dx_mat[1, 1]
        dn3_dx = dn_dx_mat[0, 2]
        dn3_dy = dn_dx_mat[1, 2]
        dn4_dx = dn_dx_mat[0, 3]
        dn4_dy = dn_dx_mat[1, 3]

        b_mat = np.array(
            [
                [dn1_dx, 0, dn2_dx, 0, dn3_dx, 0, dn4_dx, 0],
                [0, dn1_dy, 0, dn2_dy, 0, dn3_dy, 0, dn4_dy],
                [dn1_dy, dn1_dx, dn2_dy, dn2_dx, dn3_dy, dn3_dx, dn4_dy, dn4_dx],
            ]
        )

        return b_mat

    def evaluate_jacob_determinant_at(self, int_point: Point):
        jacobian = self._evaluate_jacob_at(int_point)
        return np.linalg.det(jacobian)

    def _evaluate_jacob_at(self, int_point: Point):
        _, dn_mat = self._get_shape_functions_and_their_derivatives(int_point)

        jacobian = dn_mat.dot(self.nodes)
        return jacobian

    def _get_shape_functions_and_their_derivatives(self, point: Point):
        # Note that the shape function matrix has two rows and four columns.
        # Each column is associated with a node. Each row is associated with x or y direction.
        # For instance, the value at (1,0) is the shape function of the second node associated with
        # the displacement (or force) in the x direction

        n1 = 0.25 * (1 - point.xi) * (1 - point.eta)
        n2 = 0.25 * (1 + point.xi) * (1 - point.eta)
        n3 = 0.25 * (1 + point.xi) * (1 + point.eta)
        n4 = 0.25 * (1 - point.xi) * (1 + point.eta)

        n_mat = np.array([n1, n2, n3, n4])

        dn1_dxi = -0.25 * (1 - point.eta)
        dn1_deta = -0.25 * (1 - point.xi)
        dn2_dxi = 0.25 * (1 - point.eta)
        dn2_deta = -0.25 * (1 + point.xi)
        dn3_dxi = 0.25 * (1 + point.eta)
        dn3_deta = 0.25 * (1 + point.xi)
        dn4_dxi = -0.25 * (1 + point.eta)
        dn4_deta = 0.25 * (1 - point.xi)

        dn_dxi_mat = np.array(
            [
                [dn1_dxi, dn2_dxi, dn3_dxi, dn4_dxi],
                [dn1_deta, dn2_deta, dn3_deta, dn4_deta],
            ]
        )

        return n_mat, dn_dxi_mat


def get_shape_function(nodes, element_type):
    if element_type == "Q4":
        return Q4ShapeFunction(nodes)

    elif element_type == "L2":
        return L2ShapFunction(nodes)


if __name__ == "__main__":
    point = Point(0, 0)

    nodes = np.array(
        [
            [0, 0],
            [5, 0],
            [5, 5],
            [0, 5],
        ]
    )
    # print(Q4ShapeFunction(nodes).evaluate_b_at(point))
    # print(Q4ShapeFunction(nodes)._evaluate_jacob_at(point))
    print(
        L2ShapFunction(np.array([[0, 0], [1, 0]])).evaluate_jacob_determinant_at(point)
    )
