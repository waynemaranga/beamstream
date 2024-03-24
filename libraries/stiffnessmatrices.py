# Module for stiffness matrices
import numpy as np


# --- << Member Stiffness Matrices
def stiffness_matrices(E_I, L):
    k_iij = np.multiply(
        np.array(
            [
                [(12.0 / (L**3.0)), (-6.0 / (L**2.0))],
                [(-6.0 / (L**2.0)), (4.0 / L)],
            ]
        ),
        E_I,
    )

    k_ij = np.multiply(
        np.array(
            [
                [(12.0 / (L**3.0)), (-6.0 / (L**2.0))],
                [(-6.0 / (L**2.0)), (2.0 / L)],
            ]
        ),
        E_I,
    )

    k_jji = k_iij
    k_ji = k_ij

    return [k_iij, k_ij, k_jji, k_ji]
