# --- Package imports
from utilities.interpoleit import interpoleit
import libraries.betatable as betas
from libraries.betatable import bound_list, slab_def_map


class Slab:
    def __init__(self, lx, ly):
        self.lx = lx
        self.ly = ly
        self.ly_lx = ly / lx

    def area(self):
        self.area = self.ly * self.lx
        return self.area

    def actions(self, actions):
        self.actions = actions
        return actions

    # --- Beta
    def beta_sy(self, panel_type):
        slab_type = slab_def_map[panel_type.upper()]

        beta_sy = betas.beta[slab_type]["long"]
        if beta_sy:
            beta_sy_neg_M, beta_sy_pos_M = beta_sy
            self.beta_sy_neg_M, self.beta_sy_pos_M = beta_sy
            return [beta_sy_neg_M, beta_sy_pos_M]

    def beta_sx(self, panel_type):
        slab_type = slab_def_map[panel_type.upper()]
        for i in range(len(bound_list) - 1):
            if bound_list[i] <= self.ly_lx <= bound_list[i + 1]:
                lower_bound = bound_list[i]
                upper_bound = bound_list[i + 1]

                lower_pos_M = betas.beta[slab_type][str(lower_bound)][1]
                upper_neg_M = betas.beta[slab_type][str(upper_bound)][0]
                lower_neg_M = betas.beta[slab_type][str(lower_bound)][0]
                upper_pos_M = betas.beta[slab_type][str(upper_bound)][1]
                beta_sx_neg_M = interpoleit(
                    lower_bound, upper_bound, self.ly_lx, lower_neg_M, upper_neg_M
                )
                beta_sx_pos_M = interpoleit(
                    lower_bound, upper_bound, self.ly_lx, lower_pos_M, upper_pos_M
                )
                self.beta_sx_neg_M = beta_sx_neg_M
                self.beta_sx_pos_M = beta_sx_pos_M

                return [round(beta_sx_neg_M, 4), round(beta_sx_pos_M, 4)]

    def moments(self):
        # short span
        m_sx_support = -self.beta_sx_neg_M * self.actions * self.lx**2.0
        m_sx_midspan = self.beta_sx_pos_M * self.actions * self.lx**2.0

        # long span
        m_sy_support = -self.beta_sy_neg_M * self.actions * self.lx**2.0
        m_sy_midspan = self.beta_sy_pos_M * self.actions * self.lx**2.0

        self.m_sx = [m_sx_support, m_sx_midspan]
        self.m_sy = [m_sy_support, m_sy_midspan]
        return [self.m_sx, self.m_sy]

    def transfer_areas(self):
        area_triangular = 0.25 * (self.lx**2)
        area_trapezoid = 0.5 * ((self.lx * self.ly) - (self.lx**2))
        self.triangular = area_triangular
        self.trapezoid = area_trapezoid
        return [area_triangular, area_trapezoid]

    def load_transfer(self, slab_load):
        self.transfer_areas()
        load_triangular = slab_load * self.triangular
        load_trapezoid = slab_load * self.trapezoid
        return [load_triangular, load_trapezoid]
