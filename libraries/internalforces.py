"""# Internal member end forces"""

# pylint: disable = C0103

from typing import Any
import numpy as np
from numpy import ndarray, dtype


def internal_forces(stiffnesses, betas, deltas, F_ijF) -> list[Any]:
    """Internal Forces and Moments"""
    # stiffnesses and betas are lists of two matrices. Each internal force vector needs one of each
    #   - [k_iij, k_ij] and [beta_ij, beta_ji]
    k_iij = stiffnesses[0]
    k_ij = stiffnesses[1]
    beta_ij = betas[0]
    beta_ji = betas[1]

    # F_ijF is a column vector

    # each force vector needs four displacement variables, in twos
    # deltas is a list of 4, to be accessed with indexing
    # [F_ij] = [k_iij][beta_ij][delta_i] + [k_ij][beta_ji][delta_j] + [F_ijF]
    delta_i: ndarray[Any, dtype[Any]] = np.array(
        object=[[deltas[0][0]], [deltas[1][0]]]
    )
    delta_j: ndarray[Any, dtype[Any]] = np.array(
        object=[[deltas[2][0]], [deltas[3][0]]]
    )

    F_ij: Any = np.add(
        np.matmul(k_iij, np.matmul(beta_ij, delta_i)),
        np.add(np.matmul(k_ij, np.matmul(beta_ji, delta_j)), F_ijF),
    )

    return [F_ij[0], F_ij[1]]
