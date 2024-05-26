"""Beam Analysis"""

# pylint: disable = C0103
# pylint: disable = C0301
# pylint: disable = W0201
# pylint: disable = W0511:fixme

# --- Python Packages
from typing import Any, Union
from operator import neg
import numpy as np
from numpy import ndarray, dtype
import pandas as pd
from pandas import DataFrame

# --- User-defined
from libraries.internalforces import internal_forces
from libraries.fixedendforces import fixed_end_forces
from libraries.equilibriummatrices import (beta_12, beta_12_T, beta_21, beta_21_T, beta_23, beta_23_T, beta_32, beta_32_T)  # fmt: skip
from libraries.stiffnessmatrices import stiffness_matrices

# --- Degrees of Freedom and Displacement variables ---
[d1, d2, d3, d4, d5, d6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
delta_1: ndarray[Any, dtype[Any]] = np.array(object=[[d1], [d2], [d3]])  # displaceable
delta_2: ndarray[Any, dtype[Any]] = np.array(object=[[d4], [d5], [d6]])  # restrained

Number = Union[float, int]


def round2(i) -> float:
    """Round off to 2 digits"""
    return round(number=i, ndigits=2)


class BeamAnalysis:
    """Beam Analysis by SSM and MDM"""

    def __init__(
        self, spans: list[Number], udls: list[Number], EI_values: list[Number] = [1, 1]
    ) -> None:
        # FIXME: Dangerous default value [] as argument - PylintW0102:dangerous-default-value
        self.span_12 = spans[0]
        self.span_23 = spans[1]
        self.udl_12 = udls[0]
        self.udl_23 = udls[1]
        self.EI_12 = EI_values[0]
        self.EI_23 = EI_values[1]

    def fefs(self):
        """Fixed End Forces"""
        # --- Fixed struture forces ---
        udl_12 = self.udl_12
        udl_23 = self.udl_23
        L_12 = self.span_12
        L_23 = self.span_23
        EI_12 = self.EI_12
        EI_23 = self.EI_23

        # --- Span 1-2
        V12_F, M12_F, V21_F, M21_F, delta_12 = fixed_end_forces(
            udl=udl_12,
            span=L_12,
            EI=EI_12,
        )
        F_12F: ndarray[Any, dtype[Any]] = np.array(object=[[V12_F], [M12_F]])
        F_21F: ndarray[Any, dtype[Any]] = np.array(object=[[V21_F], [M21_F]])

        # --- Span 2-3
        V23_F, M23_F, V32_F, M32_F, delta_23 = fixed_end_forces(
            udl=udl_23,
            span=L_23,
            EI=EI_23,
        )
        F_23F: ndarray[Any, dtype[Any]] = np.array(object=[[V23_F], [M23_F]])
        F_32F: ndarray[Any, dtype[Any]] = np.array(object=[[V32_F], [M32_F]])

        self.F12_F = F_12F
        self.F21_F = F_21F
        self.F23_F = F_23F
        self.F32_F = F_32F

        # return [F_12F, F_21F, F_23F, F_32F] # for testing

    def tejls(self):
        """Transformed Equivalent Joint Loads"""
        F12_F: None | ndarray[Any, dtype[Any]] = self.F12_F
        F21_F: None | ndarray[Any, dtype[Any]] = self.F21_F
        F23_F: None | ndarray[Any, dtype[Any]] = self.F23_F
        F32_F: None | ndarray[Any, dtype[Any]] = self.F32_F

        # Equivalent joint loads
        # --- Equivalent Joint Loads ---
        P1_E = np.matmul(beta_12_T, F12_F)  # for d_4 and d_1
        P2_E = np.add(
            np.matmul(beta_21_T, F21_F), np.matmul(beta_23_T, F23_F)
        )  # for d_5 and d_2
        P3_E = np.matmul(beta_32_T, F32_F)  # for d_6 and d_3

        [d_4, d_1] = P1_E  # shear force at [0], moment at [1]
        [d_5, d_2] = P2_E
        [d_6, d_3] = P3_E

        # --- Transformed Equivalent Joint Loads ---
        P1_A: ndarray[Any, dtype[Any]] = np.array(
            object=[[0.0], [0.0], [0.0]]
        )  # CLARIFY: should be support yields etc
        P1_E_T: ndarray[Any, dtype[Any]] = np.array(
            object=[[d_1[0]], [d_2[0]], [d_3[0]]]
        )
        P1: ndarray[Any, dtype[Any]] = np.subtract(P1_A, P1_E_T)

        P2: ndarray[Any, dtype[Any]] = np.array(
            object=[[0.0], [0.0], [0.0], [0.0]]
        )  # corresp. to reactions, V1, V2, V3 i.e restrained DOFs
        P2_E_T: ndarray[Any, dtype[Any]] = np.array(
            object=[[d_4[0]], [d_5[0]], [d_6[0]]]
        )

        self.P1 = P1
        self.P2 = P2
        self.P2_E_T = P2_E_T

    def stiffnesses(self) -> None:
        """Member Stiffness Matrices"""
        member_12: list[ndarray[Any, dtype[Any]]] = stiffness_matrices(
            E_I=self.EI_12, L=self.span_12
        )
        member_23: list[ndarray[Any, dtype[Any]]] = stiffness_matrices(
            E_I=self.EI_23, L=self.span_23
        )

        k_112: ndarray[Any, dtype[Any]] = member_12[0]  # k_iij
        k_12: ndarray[Any, dtype[Any]] = member_12[1]  # k_ij
        k_221: ndarray[Any, dtype[Any]] = member_12[2]  # k_jji
        k_21: ndarray[Any, dtype[Any]] = member_12[3]  # k_ji

        k_223: ndarray[Any, dtype[Any]] = member_23[0]  # k_ii
        k_23: ndarray[Any, dtype[Any]] = member_23[1]  # k_ij
        k_332: ndarray[Any, dtype[Any]] = member_23[2]  # k_jji
        k_32: ndarray[Any, dtype[Any]] = member_23[3]  # k_ji

        # --- Direct Transformed Stiffness Matrices ---
        K_112 = np.matmul(beta_12_T, np.matmul(k_112, beta_12))
        K_221 = np.matmul(beta_21_T, np.matmul(k_221, beta_21))

        K_223 = np.matmul(beta_23_T, np.matmul(k_223, beta_23))
        K_332 = np.matmul(beta_32_T, np.matmul(k_332, beta_32))

        # --- Cross Transformed Stiffness Matrices ---
        K_12 = np.matmul(beta_12_T, np.matmul(k_12, beta_21))
        K_21 = np.matmul(beta_21_T, np.matmul(k_21, beta_12))

        K_23 = np.matmul(beta_23_T, np.matmul(k_23, beta_32))
        K_32 = np.matmul(beta_32_T, np.matmul(k_32, beta_23))

        # --- Structure Stiffness Matrix ---
        ssm: ndarray = np.zeros(
            shape=(6, 6)
        )  # initialise depending on contents of delta_1 and delta_2, and other initialisations

        # --- << S11
        ssm[0, 0] = K_112[1, 1]  # d_1-1
        ssm[0, 1] = K_12[1, 1]  # d_1-2
        # ssm[0][2] = 0.0         # d_1-3, nodes 1 & 3, no fx

        ssm[1, 0] = ssm[0, 1]  # d_2-1 ; symmetry :smile:
        ssm[1, 1] = K_221[1, 1] + K_223[1, 1]  # d_2-2, fx from both spans
        ssm[1, 2] = K_23[1, 1]  # d_2-3

        ssm[2, 0] = ssm[0, 2]  # d_3-1 ; symmetry :smile:, also 0.0, no fx nodes 1 & 3
        ssm[2, 1] = ssm[1, 2]  # d_3-2 ; symmetry :smile:
        ssm[2, 2] = K_332[1, 1]  # d_3-3

        # --- << S12
        ssm[0, 3] = K_112[1, 0]  # d_1-4
        ssm[0, 4] = K_12[1, 0]  # d_1-5
        # ssm[0][5] = 0.0         # d_1-6, nodes 1 & 3, no fx

        ssm[1, 3] = K_21[1, 0]  # d_2-4
        ssm[1, 4] = K_221[1, 0] + K_223[1, 0]  # d_2-5, fx from both spans
        ssm[1, 5] = K_23[1, 0]  # d_2-6

        # ssm[2, 3] = 0.0  # d_3-4, nodes 1 & 3, no fx
        ssm[2, 4] = K_32[1, 0]  # d_3-5
        ssm[2, 5] = K_332[1, 0]  # d_3-6

        S_11: ndarray[Any, dtype[Any]] = ssm[:3, :3]
        S_12: ndarray[Any, dtype[Any]] = ssm[:3, 3:]
        # S_21 = S_12.transpose()
        # S_22 = ssm[3:, 3:]  # won't be used

        self.S_11: ndarray[Any, dtype[Any]] = S_11
        self.S_12: ndarray[Any, dtype[Any]] = S_12
        self.S_21: ndarray[Any, dtype[Any]] = S_12.transpose()
        # self.S_22 = S_22

        (
            self.k_112,
            self.k_12,
            self.k_221,
            self.k_21,
            self.k_223,
            self.k_23,
            self.k_332,
            self.k_32,
        ) = (k_112, k_12, k_221, k_21, k_223, k_23, k_332, k_32)

    def results(self) -> list[list[Any]]:
        """Analysis Results"""
        self.fefs()
        self.tejls()
        self.stiffnesses()

        S_11, S_12, S_21 = self.S_11, self.S_12, self.S_21
        F12_F, F21_F, F23_F, F32_F = self.F12_F, self.F21_F, self.F23_F, self.F32_F
        k_112, k_12, k_221, k_21, k_223, k_23, k_332, k_32 = (
            self.k_112,
            self.k_12,
            self.k_221,
            self.k_21,
            self.k_223,
            self.k_23,
            self.k_332,
            self.k_32,
        )

        # Nodal Displacements
        P1 = self.P1
        S_11_inverse = np.linalg.inv(a=S_11)
        delta_1 = np.matmul(S_11_inverse, P1)

        # Reactions
        P2_E_T = self.P2_E_T
        P2 = np.matmul(S_21, delta_1) + P2_E_T
        self.P2 = P2

        # --- Member End Forces ---
        # -> Displacement variables
        dis_1, dis_2, dis_3 = delta_1  # displaceable
        dis_4, dis_5, dis_6 = delta_2  # restrained

        # --- << F_12

        # [F_12] = [k_112][beta_12][dis_4, dis_1] + [k_12][beta_21][dis_5, dis_2] + [F_12F]
        F_12: list[Any] = internal_forces(
            stiffnesses=[k_112, k_12],
            betas=[beta_12, beta_21],
            deltas=[dis_4, dis_1, dis_5, dis_2],
            F_ijF=F12_F,
        )

        # --- << F_21
        # [F_21] = [k_221][beta_21][dis_5, dis_2] + [k_21][beta_12][dis_4, dis_1] + [F_21F]
        F_21: list[Any] = internal_forces(
            stiffnesses=[k_221, k_21],
            betas=[beta_21, beta_12],
            deltas=[dis_5, dis_2, dis_4, dis_1],
            F_ijF=F21_F,
        )

        # --- << F_23
        # [F_23] = [k_223][beta_23][dis_5, dis_2] + [k_23][beta_32][dis_6, dis_3] + [F_23F]
        F_23: list[Any] = internal_forces(
            stiffnesses=[k_223, k_23],
            betas=[beta_23, beta_32],
            deltas=[dis_5, dis_2, dis_6, dis_3],
            F_ijF=F23_F,
        )

        # --- << F_32
        # [F_32] = [k_332][beta_32][dis_6, dis_3] + [k_32][beta_23][dis_5, dis_2] + [F_32F]
        F_32: list[Any] = internal_forces(
            stiffnesses=[k_332, k_32],
            betas=[beta_32, beta_23],
            deltas=[dis_6, dis_3, dis_5, dis_2],
            F_ijF=F32_F,
        )

        # -- Correction for floating point zero erros in F_12 and F_32 moments
        threshold = 1e-5  # threshold value

        F_12[1] = [0.0] if abs(F_12[1]) < threshold else F_12[1]
        F_32[1] = [0.0] if abs(F_32[1]) < threshold else F_32[1]

        self.F_12, self.F_21, self.F_23, self.F_32 = F_12, F_21, F_23, F_32

        disp_1, disp_2, disp_3 = delta_1
        displacements: list[Number] = [
            round(number=disp_1[0], ndigits=4),
            round(number=disp_2[0], ndigits=4),
            round(number=disp_3[0], ndigits=4),
        ]
        self.displacements: list[Number] = displacements

        return [F_12, F_21, F_23, F_32, displacements]

    def get_results(self) -> list[list[Any]]:
        """Returns result as a nested list of ..."""
        self.results()

        # Reactions
        self.R_1 = round(number=neg(self.P2[0][0]), ndigits=2)
        self.R_2 = round(number=neg(self.P2[1][0]), ndigits=2)
        self.R_3 = round(number=neg(self.P2[2][0]), ndigits=2)

        # F_12
        self.V_12 = round(number=self.F_12[0][0], ndigits=2)
        self.M_12 = round(number=self.F_12[1][0], ndigits=2)
        # F_21
        self.V_21: float = round(number=self.F_21[0][0], ndigits=2)
        self.M_21: float = round(number=self.F_21[1][0], ndigits=2)
        # F_23
        self.V_23: float = round(number=self.F_23[0][0], ndigits=2)
        self.M_23: float = round(number=self.F_23[1][0], ndigits=2)
        # F_32
        self.V_32: float = round(number=self.F_32[0][0], ndigits=2)
        self.M_32: float = round(number=self.F_32[1][0], ndigits=2)

        # Displacements
        self.slope_1: float = round(number=self.displacements[0], ndigits=3)
        self.slope_2: float = round(number=self.displacements[1], ndigits=3)
        self.slope_3: float = round(number=self.displacements[2], ndigits=3)

        # Deflections # TODO: derive and add

        return [
            [self.R_1, self.R_2, self.R_3],
            [self.V_12, self.M_12],
            [self.V_21, self.M_21],
            [self.V_23, self.M_23],
            [self.V_32, self.M_32],
            [self.slope_1, self.slope_2, self.slope_3],
        ]

    def print_results(self) -> None:
        """Print results to std.out"""
        self.get_results()

        print("Reactions:")
        print(f"R_1: {self.R_1} kN")
        print(f"R_2: {self.R_2} kN")
        print(f"R_3: {self.R_3} kN")

        print("Internal forces:")
        print(f"V_12: {self.V_12} kN, M_12: {self.M_12} kNm")
        print(f"V_21: {self.V_21} kN, M_21: {self.M_21} kNm")
        print(f"V_23: {self.V_23} kN, M_23: {self.M_23} kNm")
        print(f"V_32: {self.V_32} kN, M_32: {self.M_32} kNm")

        print("Nodal Displacements:")
        print(f"Slope 1: {self.slope_1} radians")
        print(f"Slope 2: {self.slope_2} radians")
        print(f"Slope 3: {self.slope_3} radians")

    def get_reactions(self) -> list[Any]:
        """Return support reactions for analysis results"""

        self.get_results()
        return [self.R_1, self.R_2, self.R_3]

    def redistribution(self, percentage) -> list[list[Any]]:
        """Returns redistributed moments at middle support and redistributed reactions"""

        self.results()

        # TODO: should recalc. reactions and shears from redistrubted moments in member-end forces
        re_factor = 1 - (
            percentage / 100
        )  # redistribution factor i.e if percentage = 15, re_factor = 0.85
        final_moments: list[Number] = [self.F_21[1], self.F_23[1]]

        lhs: float = final_moments[0] / self.span_12  # for readability
        rhs: float = final_moments[0] / self.span_23
        reactions_fx: list[float] = [neg(lhs), (lhs + rhs), neg(rhs)]
        lhs_fr, rhs_fr = (self.udl_12 * self.span_12 / 2), (
            self.udl_23 * self.span_23 / 2
        )
        reactions_fr: list[float] = [lhs_fr, (lhs_fr + rhs_fr), rhs_fr]
        reactions_fin: list[float] = [
            (reactions_fr[0] + reactions_fx[0]),
            (reactions_fr[1] + reactions_fx[1]),
            (reactions_fr[2] + reactions_fx[2]),
        ]
        # reactions_fin = [fr + fx for fr, fx in zip(reactions_fr, reactions_fx)] # using zip TODO: check this out

        # Moment redistribution
        re_moments = [(re_factor * final_moments[0]), (re_factor * final_moments[1])]
        self.re_moments = re_moments
        # new_moments = [0.85 * x for x in re_moments] # list comprehension i.e applies fn to all items in the list
        lhs_re = re_moments[0][0] / self.span_12
        rhs_re = re_moments[0][0] / self.span_23
        reactions_fx_re = [neg(lhs_re), (lhs_re + rhs_re), neg(rhs_re)]
        reactions_fin_re = [
            (reactions_fr[0] + reactions_fx_re[0]),
            (reactions_fr[1] + reactions_fx_re[1]),
            (reactions_fr[2] + reactions_fx_re[2]),
        ]

        self.reactions_fin: list[float] = (
            reactions_fin  # final reactions before redistribution
        )
        self.reactions_fin_re: list[Number] = (
            reactions_fin_re  # final reactions after redistribution
        )

        return [self.re_moments, self.reactions_fin_re]


class MomentDistribution(BeamAnalysis):
    """Beam Analysis by the moment distribution method"""

    def __init__(self, spans, udls, EI_values=[1, 1]) -> None:
        super().__init__(spans=spans, udls=udls, EI_values=EI_values)
        zeroes: ndarray = np.zeros(shape=(13, 4))  # numpy array with 13 rows and 4 columns of zeroes # fmt: skip
        columns: list[str] = ["AB", "BA", "BC", "CB"]  # columns
        idxs: list[str] = [
            "STIFF",  # Stiffnesses
            "DF",  # Distribution Factors
            "FEM",  # Fixed-End Moments
            "BAL",  # Balance
            "COM",  # Carry-Over Moments
            "BAL",  # Balance
            "COM",  # Carry-Over
            "BAL",  # Balance
            "FMs",  # Final Moment
            "ReFMs",  # Redistributed Final Moments
            "FxRx",  # Fixed Reactions
            "FrRx",  # Free Reactions
            "FnRx",  # Final/Resultant Reaction
        ]  # rows
        self.df = pd.DataFrame(data=zeroes, columns=columns, index=idxs)

        pd.set_option(
            "display.precision", 3
        )  # can't by typed, function must have an even no. of non-keyword arguments, ValueError for pd.set_option
        self.L_12 = spans[0]
        self.L_23 = spans[1]
        self.w_12 = udls[0]
        self.w_23 = udls[1]

        # self.re_f = 0.85 # FIXME: default value, should be user-defined

    def run(self, redistribution) -> None:
        """Run Analysis"""
        # ->
        df: DataFrame = self.df
        self.re_f = (
            100 - redistribution
        ) / 100  # FIXME: default value, should be user-defined

        # --- Fixed End Moments ---
        df.at["FEM", "AB"] = neg(self.w_12 * self.L_12**2 / 12)  # joint A
        df.at["FEM", "BA"] = self.w_12 * self.L_12**2 / 12  # joint B, span 1-2
        df.at["FEM", "BC"] = neg(self.w_23 * self.L_23**2 / 12)  # joint B, span 2-3
        df.at["FEM", "CB"] = self.w_23 * self.L_23**2 / 12  # joint C

        # --- Stiffnesses ---
        df.at["STIFF", "AB"] = 0.75 / self.L_12  # 3EI/4L for pinned end
        stiff_12 = df.at["STIFF", "AB"]  # redundant
        df.at["STIFF", "BA"] = 0.75 / self.L_12
        df.at["STIFF", "BC"] = 0.75 / self.L_23
        stiff_23 = df.at["STIFF", "BC"]  # redundant
        df.at["STIFF", "CB"] = 0.75 / self.L_23

        # --- Distribution Factors ---
        df.at["DF", "AB"] = 1.0  # pinned end has DF of 1.0
        df.at["DF", "CB"] = 1.0
        df.at["DF", "BA"] = stiff_12 / (stiff_12 + stiff_23)
        df.at["DF", "BC"] = stiff_23 / (stiff_12 + stiff_23)

        # --- Try to display pd.DataFrame in Streamlit with Animation ---
        # --- Balancing... ---
        # -> 1st iteration...
        df.iat[3, 0] = neg(df.iat[2, 0])  # BAL at joint A
        df.iat[3, 3] = neg(df.iat[2, 3])  # BAL at joint C
        obm_i = df.iat[2, 1] + df.iat[2, 2]  # OMB at joint B, 1st iter
        df.iat[3, 1] = neg(obm_i * df.iat[1, 1])  # BAL at BA
        df.iat[3, 2] = neg(obm_i * df.iat[1, 2])  # BAL at BC
        df.iat[4, 0] = df.iat[3, 1] / 2  # COM at joint A
        df.iat[4, 3] = df.iat[3, 2] / 2  # COM at joint C
        df.iat[4, 1] = df.iat[3, 0] / 2  # COM at BA
        df.iat[4, 2] = df.iat[3, 3] / 2  # COM at BC

        # -> 2nd iteration...
        obm_ii = df.iat[4, 1] + df.iat[4, 2]  # OMB at joint B, 2nd iter
        df.iat[5, 0] = neg(df.iat[4, 0])  # BAL at joint A
        df.iat[5, 3] = neg(df.iat[4, 3])  # BAL at joint C
        df.iat[5, 1] = neg(obm_ii * df.iat[1, 1])  # BAL at BA
        df.iat[5, 2] = neg(obm_ii * df.iat[1, 2])  # BAL at BC
        df.iat[6, 0] = df.iat[5, 1] / 2  # COM at joint A
        df.iat[6, 3] = df.iat[5, 2] / 2  # COM at joint C
        df.iat[6, 1] = df.iat[5, 0] / 2  # COM at BA
        df.iat[6, 2] = df.iat[5, 3] / 2  # COM at BC

        # -> 3rd iteration...
        obm_iii = df.iat[6, 1] + df.iat[6, 2]  # OMB at joint B, 3rd iter
        df.iat[7, 0] = neg(df.iat[6, 0])  # BAL at joint A
        df.iat[7, 3] = neg(df.iat[6, 3])  # BAL at joint C
        df.iat[7, 1] = neg(obm_iii * df.iat[1, 1])  # BAL at BA
        df.iat[7, 2] = neg(obm_iii * df.iat[1, 2])  # BAL at BC

        # --- Moments, pre-redist ---
        df.iat[8, 0] = df.iloc[2:8, 0].sum()  # FM @ joint A
        df.iat[8, 3] = df.iloc[2:8, 3].sum()  # FM @ joint C
        df.iat[8, 1] = df.iloc[2:8, 1].sum()  # FM @ BA
        df.iat[8, 2] = df.iloc[2:8, 2].sum()  # FM @ BC

        self.M_12_no_dist = (
            0  # round(df.iat[8,0], 2) # forced to 0 to avoid floating pt errors
        )
        self.M_21_no_dist = round(df.iat[8, 1], 2)
        self.M_23_no_dist = round(df.iat[8, 2], 2)
        self.M_32_no_dist = 0  # round(df.iat[8,3], 2)

        # --- Moments, post-redist. ---
        df.iat[9, 0] = (
            0  # Joint A # force set to 0 (for pinned end) to avoid floating pt errors
        )
        df.iat[9, 3] = 0  # Joint C
        df.iat[9, 1] = df.iat[8, 1] * self.re_f  # redist. FM at BA
        df.iat[9, 2] = df.iat[8, 2] * self.re_f  # redist. FM at CB

        self.M_12_dist = (
            0  # round(df.iat[9,0]) # joint A, forced to 0 to avoid floating pt errors
        )
        self.M_21_dist = round(df.iat[9, 1])  # joint B,
        self.M_23_dist = round(df.iat[9, 2])  # joint B,
        self.M_32_dist = 0  # round(df.iat[9,3]) # joint C

        # --- Fixed Reactions
        df.iat[10, 0] = neg(df.iat[9, 1]) / self.L_12  # joint A
        df.iat[10, 1] = (df.iat[9, 1]) / self.L_12  # joint B, span 1-2
        df.iat[10, 2] = neg((df.iat[9, 2])) / self.L_23  # joint B, span 2-3
        df.iat[10, 3] = (df.iat[9, 2]) / self.L_23  # joint C

        # --- Free Reactions
        df.iat[11, 0] = self.w_12 * self.L_12 / 2  # joint A
        df.iat[11, 1] = self.w_12 * self.L_12 / 2  # joint B, span 1-2
        df.iat[11, 2] = self.w_23 * self.L_23 / 2  # joint B, span 2-3
        df.iat[11, 3] = self.w_23 * self.L_23 / 2  # joint C

        # --- Final Reactions
        df.iat[12, 0] = df.iat[10, 0] + df.iat[11, 0]  # joint A
        df.iat[12, 1] = df.iat[10, 1] + df.iat[11, 1]  # joint B, span 1-2
        df.iat[12, 2] = df.iat[10, 2] + df.iat[11, 2]  # joint B, span 2-3
        df.iat[12, 3] = df.iat[10, 3] + df.iat[11, 3]  # joint C

        self.R_1_dist = round2(i=df.iat[12, 0])  # joint A
        self.R_2_dist = round2(i=df.iat[12, 1] + df.iat[12, 2])  # joint B
        self.R_3_dist = round2(i=df.iat[12, 3])  # joint C

        self.df_final = df

        # --- Calc reactions pre-redist ---
        # -- Fixed reactions, pre-redist ---
        R_1fx = neg(df.iat[8, 1] / self.L_12)  # joint A
        R_2fx = neg(R_1fx) + neg(df.iat[8, 2] / self.L_23)  # joint B
        R_3fx = df.iat[8, 2] / self.L_23  # joint C

        # -- Final reactions, pre-redist. (Free reactions already calc.d)
        self.R_1_nodist = round2(i=R_1fx + df.iat[11, 0])
        self.R_2_nodist = round2(i=R_2fx + df.iat[11, 1] + df.iat[11, 2])  #
        self.R_3_nodist = round2(i=R_3fx + df.iat[11, 3])

        # return df
        # + if-elif-else for runs/errors & exceptions for diff't return types #TODO: try this
        self.md_df: str = f"""
|          | AB                      | BA                      | BC                      | CB                      |
| -------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| STIFF    | {round2(i=df.iat[0,0])}   | {round2(i=df.iat[0,1])}   | {round2(i=df.iat[0,2])}   | {round2(i=df.iat[0,3])}   |
| DF       | {round2(i=df.iat[1,0])}   | {round2(i=df.iat[1,1])}   | {round2(i=df.iat[1,2])}   | {round2(i=df.iat[1,3])}   |
| FEM      | {round2(i=df.iat[2,0])}   | {round2(i=df.iat[2,1])}   | {round2(i=df.iat[2,2])}   | {round2(i=df.iat[2,3])}   |
| BAL      | {round2(i=df.iat[3,0])}   | {round2(i=df.iat[3,1])}   | {round2(i=df.iat[3,2])}   | {round2(i=df.iat[3,3])}   |
| COM      | {round2(i=df.iat[4,0])}   | {round2(i=df.iat[4,1])}   | {round2(i=df.iat[4,2])}   | {round2(i=df.iat[4,3])}   |
| BAL      | {round2(i=df.iat[5,0])}   | {round2(i=df.iat[5,1])}   | {round2(i=df.iat[5,2])}   | {round2(i=df.iat[5,3])}   |
| COM      | {round2(i=df.iat[6,0])}   | {round2(i=df.iat[6,1])}   | {round2(i=df.iat[6,2])}   | {round2(i=df.iat[6,3])}   |
| BAL      | {round2(i=df.iat[7,0])}   | {round2(i=df.iat[7,1])}   | {round2(i=df.iat[7,2])}   | {round2(i=df.iat[7,3])}   |
| FMs      | {round2(i=df.iat[8,0])}   | {round2(i=df.iat[8,1])}   | {round2(i=df.iat[8,2])}   | {round2(i=df.iat[8,3])}   |
| ReFMs    | {round2(i=df.iat[9,0])}   | {round2(i=df.iat[9,1])}   | {round2(i=df.iat[9,2])}   | {round2(i=df.iat[9,3])}   |
| FxRx     | {round2(i=df.iat[10,0])}  | {round2(i=df.iat[10,1])}  | {round2(i=df.iat[10,2])}  | {round2(i=df.iat[10,3])}  |
| FrRx     | {round2(i=df.iat[11,0])}  | {round2(i=df.iat[11,1])}  | {round2(i=df.iat[11,2])}  | {round2(i=df.iat[11,3])}  |
| FnRx     | {round2(i=df.iat[12,0])}  | {round2(i=df.iat[12,1])}  | {round2(i=df.iat[12,2])}  | {round2(i=df.iat[12,3])}  |
"""

    # def get_results(self):
    #     return super().get_results() #TODO: FIXME:

    def get_results(self, redistribution: Number) -> list[list[Any]]:
        """Returns a nested list of:
        \n`[moments before redist.]`,
        \n`[redistributed moments]`,
        \n`[reactions before redistr.]` and
        \n`[reactions after redistr.]`"""

        self.run(redistribution=redistribution)
        moments_pre_dist: list[Number] = [
            self.M_12_no_dist,
            self.M_21_no_dist,
            self.M_23_no_dist,
            self.M_32_no_dist,
        ]  #
        moments_redist: list[Number] = [
            self.M_12_dist,
            self.M_21_dist,
            self.M_23_dist,
            self.M_32_dist,
        ]
        reactions_pre_dist: list[float] = [
            self.R_1_nodist,
            self.R_2_nodist,
            self.R_3_nodist,
        ]
        reactions_redist: list[float] = [self.R_1_dist, self.R_2_dist, self.R_3_dist]

        return [moments_pre_dist, moments_redist, reactions_pre_dist, reactions_redist]

    def print_results(self, redistr_pct: Number) -> None:
        self.run(redistribution=redistr_pct)
        # fmt: off
        print(f"Moments (pre-redistr.): \n{[self.M_12_no_dist,self.M_21_no_dist,self.M_23_no_dist,self.M_32_no_dist]}")
        print(f"Moments (redistr.): \n{[self.M_12_dist,self.M_21_dist,self.M_23_dist,self.M_32_dist]}")
        print(f"Reactions (pre-redistr.): \n{[self.R_1_nodist,self.R_2_nodist,self.R_3_nodist]}")
        print(f"Reactions (redistr.): \n{[self.R_1_dist,self.R_2_dist,self.R_3_dist]}")
        # fmt: on

    def get_table(self) -> DataFrame:
        """Returns moment distribution tables as a `pandas.DataFrame`"""
        return self.df_final

    def get_md_table(self) -> str:
        """Returns moment distribution table as a markdown formatted string"""
        return self.md_df

    def get_results_st(self, redistr_pct: Number) -> str:
        """Returns results for streamlit frontend"""
        self.run(redistribution=redistr_pct)
        results_str: str = (
            f"Moments (pre-redistr.):  \n{[self.M_12_no_dist,self.M_21_no_dist,self.M_23_no_dist,self.M_32_no_dist]}  \nMoments (redistr.):  \n{[self.M_12_dist,self.M_21_dist,self.M_23_dist,self.M_32_dist]}  \nReactions (pre-redistr.):  \n{[self.R_1_nodist,self.R_2_nodist,self.R_3_nodist]}  \nReactions (redistr.):  \n{[self.R_1_dist,self.R_2_dist,self.R_3_dist]}"
        )
        return results_str
