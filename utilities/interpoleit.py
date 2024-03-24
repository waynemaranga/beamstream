# Module for doing interpolation
# Specifically made for beta values


# Interpolate for x
def interpoleit(y1, y2, y3, x1, x2):
    """
    Formula: `(y2 - y1)/(x2 - x1) = (y3 - y2)/(x3 - x2)`
    Returns `x3` given `y1`, `y2`, `y3`, `x1`, `x2`
    """
    x3 = (x2 - x1) / (y2 - y1) * (y3 - y2) + x2
    return x3
