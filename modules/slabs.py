"""Slabs"""

# pylint: disable = C0301
# pylint: disable = C0103
# pylint: disable = W0201

# --- Package imports
from typing import Any
from utilities.interpoleit import interpoleit
import libraries.betatable as betas
from libraries.betatable import bound_list, slab_def_map


class Slab:
    """Two-way RC Slab"""

    def __init__(self, lx: float | int, ly: float | int) -> None:
        self.lx: float | int = lx
        self.ly: float | int = ly
        self.ly_lx: float = ly / lx

    # def area(self): # Method declaration "area" is obscured by a declaration of the same name PylancereportRedeclaration
    def area(self) -> float | int:
        """Slab Area, length * width, in m2"""
        self._area: float | int = self.ly * self.lx
        return self._area

    def actions(self, actions: float | int) -> float | int:
        """Slab Actions in kN/m2"""
        self._actions: float | int = actions
        return actions

    # --- Beta
    # def beta_sy(self, panel_type) -> list[Any]: # Function with declared return type "list[Any]" must return value on all code paths
    def beta_sy(self, panel_type: str) -> list[Any] | None:
        """Beta Coeff."""
        slab_type: str = slab_def_map[panel_type.upper()]

        beta_sy: list | tuple = betas.beta[slab_type]["long"]
        if beta_sy:
            beta_sy_neg_M, beta_sy_pos_M = beta_sy
            self.beta_sy_neg_M, self.beta_sy_pos_M = beta_sy
            return [beta_sy_neg_M, beta_sy_pos_M]

    def beta_sx(self, panel_type) -> list[float] | None:
        """Beta Coeff."""

        slab_type: str = slab_def_map[panel_type.upper()]
        for i in range(len(bound_list) - 1):
            if bound_list[i] <= self.ly_lx <= bound_list[i + 1]:
                lower_bound: float = bound_list[i]
                upper_bound: float = bound_list[i + 1]

                lower_pos_M: float = betas.beta[slab_type][str(object=lower_bound)][1]
                upper_neg_M: float = betas.beta[slab_type][str(object=upper_bound)][0]
                lower_neg_M: float = betas.beta[slab_type][str(object=lower_bound)][0]
                upper_pos_M: float = betas.beta[slab_type][str(object=upper_bound)][1]
                beta_sx_neg_M: float = interpoleit(
                    y1=lower_bound,
                    y2=upper_bound,
                    y3=self.ly_lx,
                    x1=lower_neg_M,
                    x2=upper_neg_M,
                )
                beta_sx_pos_M: float = interpoleit(
                    y1=lower_bound,
                    y2=upper_bound,
                    y3=self.ly_lx,
                    x1=lower_pos_M,
                    x2=upper_pos_M,
                )
                self.beta_sx_neg_M: float = beta_sx_neg_M
                self.beta_sx_pos_M: float = beta_sx_pos_M

                return [
                    round(number=beta_sx_neg_M, ndigits=4),
                    round(number=beta_sx_pos_M, ndigits=4),
                ]

    def moments(self) -> list[list[float]]:
        """Slabs bending moments, per axis"""
        # short span
        m_sx_support: float = -self.beta_sx_neg_M * self._actions * self.lx**2.0
        m_sx_midspan: float = self.beta_sx_pos_M * self._actions * self.lx**2.0

        # long span
        m_sy_support: float = -self.beta_sy_neg_M * self._actions * self.lx**2.0
        m_sy_midspan: float = self.beta_sy_pos_M * self._actions * self.lx**2.0

        self.m_sx: list[float] = [m_sx_support, m_sx_midspan]
        self.m_sy: list[float] = [m_sy_support, m_sy_midspan]
        return [self.m_sx, self.m_sy]

    def transfer_areas(self) -> list[float]:
        """45-degree from corner transfer methods"""
        area_triangular: float = 0.25 * (self.lx**2)
        area_trapezoid: float = 0.5 * ((self.lx * self.ly) - (self.lx**2))
        self.triangular: float = area_triangular
        self.trapezoid: float = area_trapezoid
        return [area_triangular, area_trapezoid]

    def load_transfer(self, slab_load: float) -> list[float]:
        """Loads transferred to slab"""
        self.transfer_areas()
        load_triangular: float = slab_load * self.triangular
        load_trapezoid: float = slab_load * self.trapezoid
        return [load_triangular, load_trapezoid]
