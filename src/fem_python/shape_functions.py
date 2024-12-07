from abc import ABC, abstractmethod

import numpy as np

from fem_python import config


class ShapeFunction(ABC):
    def __init__(self, nodes):
        self.nodes = nodes

    @abstractmethod
    def evaluate_n_at():
        pass

    @abstractmethod
    def evaluate_b_at():
        pass

    @abstractmethod
    def _deriv_n_at():
        pass

    def evaluate_jacob_at(self, point):
        deriv_n = self._deriv_n_at(point)
        return deriv_n.dot(self.nodes)[0]


class LinearShapeFunction(ShapeFunction):
    def __init__(self, nodes):
        super().__init__(nodes)

    @staticmethod
    def evaluate_n_at(point):
        return np.array([[point - 0.5, point + 0.5]])

    @staticmethod
    def _deriv_n_at(point):
        return np.array([[-0.5, 0.5]])

    def evaluate_b_at(self, point):
        jacob = self.evaluate_jacob_at(point)
        inv_jacob = 1 / jacob
        deriv_n = self._deriv_n_at(point)
        return inv_jacob * deriv_n


class QuadShapeFunction(ShapeFunction):
    def __init__(self, nodes):
        super().__init__(nodes)

    @staticmethod
    def evaluate_n_at(point):
        return np.array(
            [[0.5 * point * (point - 1), 1 - point**2, 0.5 * point * (point + 1)]]
        )

    @staticmethod
    def _deriv_n_at(point):
        return np.array([[point - 0.5, -2 * point, point + 0.5]])

    def evaluate_b_at(self, point):
        jacob = self.evaluate_jacob_at(point)
        inv_jacob = 1 / jacob
        deriv_n = self._deriv_n_at(point)
        return inv_jacob * deriv_n


def get_shape_function():
    if config.element_type == "linear":
        return LinearShapeFunction

    if config.element_type == "quad":
        return QuadShapeFunction


if __name__ == "__main__":
    print(QuadShapeFunction([0, 0.5, 1]).evaluate_jacob_at(0.5))
