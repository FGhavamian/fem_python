from typing import List

import numpy as np


class IntegrationPoint:
    def __init__(self, location, weight):
        self.location = location
        self.weight = weight


def get_gauus_integration_setting(num_int_points) -> List[IntegrationPoint]:
    integration_points = []

    if num_int_points == 1:
        integration_point = IntegrationPoint(location=0, weight=2)
        integration_points.append(integration_point)

    elif num_int_points == 2:
        integration_point_1 = IntegrationPoint(location=-1 / np.sqrt(3), weight=1)
        integration_point_2 = IntegrationPoint(location=1 / np.sqrt(3), weight=1)

        integration_points.append(integration_point_1)
        integration_points.append(integration_point_2)

    return integration_points
