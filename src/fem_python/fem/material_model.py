import numpy as np

from fem_python import config


def get_elastic_stiffness_matrix():
    e = config.bar_elasticity_module
    nu = config.bar_poission_ratio

    if config.plane_stress:
        return (e / (1 - 2 * nu**2)) * np.array(
            [[1, nu, 0], [nu, 1, 0], [0, 0, 1 - nu]]
        )

    else:
        return (
            e
            / ((1 + nu) * (1 - 2 * nu))
            * np.array(
                [
                    [1 - nu, nu, 0],
                    [nu, 1 - nu, 0],
                    [0, 0, 1 - 2 * nu],
                ]
            )
        )


if __name__ == "__main__":
    print(get_elastic_stiffness_matrix(100, 0.1))
