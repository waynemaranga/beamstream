"""# Stiffness matrices
"""

# pylint: disable = C0103

from typing import Any, Union
import numpy as np
from numpy import dtype, ndarray

Number = Union[float, int]

# --- << Member Stiffness Matrices
def stiffness_matrices(E_I: Number, L: Number) -> list[ndarray[Any, dtype[Any]]]:
    """Member Stiffness Matrix"""
    k_iij: ndarray[Any, dtype[Any]] = np.multiply(
        np.array(
            object=[
                [(12.0 / (L**3.0)), (-6.0 / (L**2.0))],
                [(-6.0 / (L**2.0)), (4.0 / L)],
            ]
        ),
        E_I,
    )

    k_ij: ndarray[Any, dtype[Any]] = np.multiply(
        np.array(
            object=[
                [(12.0 / (L**3.0)), (-6.0 / (L**2.0))],
                [(-6.0 / (L**2.0)), (2.0 / L)],
            ]
        ),
        E_I,
    )

    k_jji = k_iij
    k_ji = k_ij

    return [k_iij, k_ij, k_jji, k_ji]
