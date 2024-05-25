"""# Module for doing interpolation
# Specifically made for beta values
"""


# Interpolate for x
def interpoleit(
    y1: float | int, y2: float | int, y3: float | int, x1: float | int, x2: float | int
) -> float | int:
    """
    Formula: `(y2 - y1)/(x2 - x1) = (y3 - y2)/(x3 - x2)`
    Returns `x3` given `y1`, `y2`, `y3`, `x1`, `x2`
    """
    x3: float = (x2 - x1) / (y2 - y1) * (y3 - y2) + x2
    return x3
