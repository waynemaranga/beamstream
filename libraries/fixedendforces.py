"""Fixed End Forces"""

# pylint: disable = C0103

from operator import neg


# Fixed-structure force function
def fixed_end_forces(
    udl: float | int, span: float | int, EI: float | int
) -> list[float]:
    """Fixed End Forces"""
    VAB_F: float = udl * span / 2
    MAB_F: float = neg(udl) * span**2 / 12

    VBA_F: float = neg(udl) * span / 2
    MBA_F: float = udl * span**2 / 12

    MC: float = udl * span**2 / 24  # moment at midspan

    delta_mid: float = udl * span**4 / (384 * EI)
    return [VAB_F, MAB_F, VBA_F, MBA_F, delta_mid]
