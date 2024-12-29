from abc import ABC, abstractmethod

import numpy as np

from fem_python import config


class AbstractMaterialModel(ABC):
    """This is an abstract class for material model. It is essentially a template for
    how to build material models. It helps standardize the interface and hence, using
    different material models is easier.

    Every material model that inherits from this class must implement the update function.
    The update function updates stiffness matrix and stress vector.

    By design, it is not possible to set the stiffness matrix and stress vector directly.
    As mentioned, the update function is in charge of setting these values.

    There is a function to get the elastic stiffness matrix. Any plasticity model that
    we implemenet is going to need the elastic stiffness matrix.
    """

    def __init__(self, elasticity_module, poission_ratio):
        self.elasticity_module = elasticity_module
        self.poission_ratio = poission_ratio

        self._stiffness_matrix = None
        self._stress = None
        self._strain = None

    @abstractmethod
    def update(self, strain):
        pass

    @property
    def stiffness_matrix(self):
        return self._stiffness_matrix

    @property
    def stress(self):
        return self._stress

    @property
    def strain(self):
        return self._strain

    @property
    def elastic_stiffness(self):
        e = self.elasticity_module
        nu = self.poission_ratio

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


class LinearElasticMaterialModel(AbstractMaterialModel):
    def __init__(self, elasticity_module, poission_ratio):
        super().__init__(elasticity_module, poission_ratio)

    def update(self, strain):
        self._strain = strain
        self._stiffness_matrix = self.elastic_stiffness
        self._stress = np.dot(self.elastic_stiffness, self._strain)


class NonelinearElasticMaterialModel(AbstractMaterialModel):
    """This is a toy example to test the implementation of a nonlinear material model."""

    def __init__(self, elasticity_module, poission_ratio):
        super().__init__(elasticity_module, poission_ratio)

    def update(self, strain):
        """This is a toy example to test the implementation of a nonlinear material model.
        The stress is: 0.5 * C * (1 + strain)^2
        And the stiffness matrix is: C * diag(1 + strain)

        Note that this nonlinearity is so mild, that a linear solver also suffices."""
        self._strain = strain
        self._stress = 0.5 * np.dot(self.elastic_stiffness, np.power(self._strain, 2))
        self._stiffness_matrix = np.dot(
            self.elastic_stiffness, np.diag(1 + self._strain)
        )


def get_material_model(material_model_name, **kwargs):
    if material_model_name == "linear_elastic":
        return LinearElasticMaterialModel(**kwargs)

    elif material_model_name == "nonlinear_elastic":
        return NonelinearElasticMaterialModel(**kwargs)

    else:
        raise NotImplementedError(
            f"Material model {material_model_name} is not implemented."
        )


if __name__ == "__main__":
    mat = get_material_model(
        "nonlinear_elastic", elasticity_module=1, poission_ratio=0.3
    )
    mat.update(np.array([1, 2, 3]))
    print(mat.stiffness_matrix)
    print(mat.stress)

    mat = get_material_model("hyper_elastic", elasticity_module=1, poission_ratio=0.3)
