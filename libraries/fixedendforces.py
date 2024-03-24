from operator import neg


# Fixed-structure force function
def fixed_end_forces(udl, span, EI):
    VAB_F = udl * span / 2
    MAB_F = neg(udl) * span**2 / 12

    VBA_F = neg(udl) * span / 2
    MBA_F = udl * span**2 / 12

    MC = udl * span**2 / 24  # moment at midspan

    delta_mid = udl * span**4 / (384 * EI)
    return [VAB_F, MAB_F, VBA_F, MBA_F, delta_mid]
