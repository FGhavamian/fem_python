from typing import List

import numpy as np

from fem_python.shape_functions import Point


class IntegrationPoint:
    def __init__(self, point: Point, weight):
        self.point = point
        self.weight = weight

    def __repr__(self):
        return f"xi={self.point.xi}, eta={self.point.eta}, weigth={self.weight}"


def get_gauus_integration_setting(num_int_points) -> List[IntegrationPoint]:

    if num_int_points == 1:
        integration_points = [IntegrationPoint(Point(xi=0, eta=0), weight=2)]

    elif num_int_points == 2:
        integration_points = [
            IntegrationPoint(Point(xi=-1 / np.sqrt(3), eta=-1 / np.sqrt(3)), weight=1),
            IntegrationPoint(Point(xi=1 / np.sqrt(3), eta=-1 / np.sqrt(3)), weight=1),
            IntegrationPoint(Point(xi=1 / np.sqrt(3), eta=1 / np.sqrt(3)), weight=1),
            IntegrationPoint(Point(xi=-1 / np.sqrt(3), eta=1 / np.sqrt(3)), weight=1),
        ]

    return integration_points


print(get_gauus_integration_setting(2))
