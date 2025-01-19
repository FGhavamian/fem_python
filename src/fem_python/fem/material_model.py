from abc import ABC, abstractmethod
from copy import deepcopy

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

    def __init__(self, elasticity_module, poission_ratio, state):
        self.elasticity_module = elasticity_module
        self.poission_ratio = poission_ratio

        self.state = deepcopy(state)
        self.tmp_state = deepcopy(state)

    @abstractmethod
    def compute_stress_and_stiffness(self, increment_strain):
        """This function should return a tuple, stress vector and the stiffness matrix"""
        pass

    def save_state(self):
        # This is a way to deepcopy a dictionary in python
        # self.state = self.tmp_state copies refrence! This is not what we want
        self.state = deepcopy(self.tmp_state)

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
        state = {"stress": np.zeros((3,))}

        super().__init__(elasticity_module, poission_ratio, state=state)

    def compute_stress_and_stiffness(self, increment_strain):
        stress = self.state["stress"]
        increment_stress = np.dot(self.elastic_stiffness, increment_strain)

        stress += increment_stress

        self.tmp_state["stress"] = stress

        return self.elastic_stiffness, stress


class NonelinearElasticMaterialModel(AbstractMaterialModel):
    """This is a toy example to test the implementation of a nonlinear material model."""

    def __init__(self, elasticity_module, poission_ratio):
        state = {"stress": np.zeros((3,)), "strain": np.zeros((3,))}

        super().__init__(elasticity_module, poission_ratio, state=state)

    def compute_stress_and_stiffness(self, increment_strain):
        """This is a toy example to test the implementation of a nonlinear material model.

        Note that this nonlinearity is so mild, that a linear solver also suffices."""
        stress = self.state["stress"]
        strain = self.state["strain"]

        strain += increment_strain

        stress = np.dot(self.elastic_stiffness, 1 - np.exp(-strain))

        stiffness_matrix = np.dot(self.elastic_stiffness, np.diag(np.exp(-strain)))

        self.tmp_state["stress"] = stress
        self.tmp_state["strain"] = strain

        return stiffness_matrix, stress


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

    mat = get_material_model("hyper_elastic", elasticity_module=1, poission_ratio=0.3)
