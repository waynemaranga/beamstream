# --- Python Packages
import numpy as np
import pandas as pd
from operator import neg

# --- User-defined
from libraries.internalforces import internal_forces
from libraries.fixedendforces import fixed_end_forces
from libraries.equilibriummatrices import (
    beta_12,
    beta_12_T,
    beta_21,
    beta_21_T,
    beta_23,
    beta_23_T,
    beta_32,
    beta_32_T,
)
from libraries.stiffnessmatrices import stiffness_matrices

# --- Degrees of Freedom and Displacement variables ---
[d1, d2, d3, d4, d5, d6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
delta_1 = np.array([[d1], [d2], [d3]])  # displaceable
delta_2 = np.array([[d4], [d5], [d6]])  # restrained


def round2(i):
    return round(i, 2)


class BeamAnalysis:
    def __init__(self, spans, udls, EI_values=[1, 1]):
        self.span_12 = spans[0]
        self.span_23 = spans[1]
        self.udl_12 = udls[0]
        self.udl_23 = udls[1]
        self.EI_12 = EI_values[0]
        self.EI_23 = EI_values[1]

    def fefs(self):
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
        F_12F = np.array([[V12_F], [M12_F]])
        F_21F = np.array([[V21_F], [M21_F]])

        # --- Span 2-3
        V23_F, M23_F, V32_F, M32_F, delta_23 = fixed_end_forces(
            udl=udl_23,
            span=L_23,
            EI=EI_23,
        )
        F_23F = np.array([[V23_F], [M23_F]])
        F_32F = np.array([[V32_F], [M32_F]])

        self.F12_F = F_12F
        self.F21_F = F_21F
        self.F23_F = F_23F
        self.F32_F = F_32F

        # return [F_12F, F_21F, F_23F, F_32F] # for testing

    def tejls(self):
        F12_F = self.F12_F
        F21_F = self.F21_F
        F23_F = self.F23_F
        F32_F = self.F32_F

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
        P1_A = np.array([[0.0], [0.0], [0.0]])  # CLARIFY: should be support yields etc
        P1_E_T = np.array([[d_1[0]], [d_2[0]], [d_3[0]]])
        P1 = np.subtract(P1_A, P1_E_T)

        P2 = np.array(
            [[0.0], [0.0], [0.0], [0.0]]
        )  # corresp. to reactions, V1, V2, V3 i.e restrained DOFs
        P2_E_T = np.array([[d_4[0]], [d_5[0]], [d_6[0]]])

        self.P1 = P1
        self.P2 = P2
        self.P2_E_T = P2_E_T

    def stiffnesses(self):
        # --- Member Stiffness Matrices ---
        member_12 = stiffness_matrices(self.EI_12, self.span_12)
        member_23 = stiffness_matrices(self.EI_23, self.span_23)

        k_112 = member_12[0]  # k_iij
        k_12 = member_12[1]  # k_ij
        k_221 = member_12[2]  # k_jji
        k_21 = member_12[3]  # k_ji

        k_223 = member_23[0]  # k_ii
        k_23 = member_23[1]  # k_ij
        k_332 = member_23[2]  # k_jji
        k_32 = member_23[3]  # k_ji

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
        ssm = np.zeros(
            (6, 6)
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

        S_11 = ssm[:3, :3]
        S_12 = ssm[:3, 3:]
        # S_21 = S_12.transpose()
        # S_22 = ssm[3:, 3:]  # won't be used

        self.S_11 = S_11
        self.S_12 = S_12
        self.S_21 = S_12.transpose()
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

    def results(self):
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
        S_11_inverse = np.linalg.inv(S_11)
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
        F_12 = internal_forces(
            stiffnesses=[k_112, k_12],
            betas=[beta_12, beta_21],
            deltas=[dis_4, dis_1, dis_5, dis_2],
            F_ijF=F12_F,
        )

        # --- << F_21
        # [F_21] = [k_221][beta_21][dis_5, dis_2] + [k_21][beta_12][dis_4, dis_1] + [F_21F]
        F_21 = internal_forces(
            stiffnesses=[k_221, k_21],
            betas=[beta_21, beta_12],
            deltas=[dis_5, dis_2, dis_4, dis_1],
            F_ijF=F21_F,
        )

        # --- << F_23
        # [F_23] = [k_223][beta_23][dis_5, dis_2] + [k_23][beta_32][dis_6, dis_3] + [F_23F]
        F_23 = internal_forces(
            stiffnesses=[k_223, k_23],
            betas=[beta_23, beta_32],
            deltas=[dis_5, dis_2, dis_6, dis_3],
            F_ijF=F23_F,
        )

        # --- << F_32
        # [F_32] = [k_332][beta_32][dis_6, dis_3] + [k_32][beta_23][dis_5, dis_2] + [F_32F]
        F_32 = internal_forces(
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
        displacements = [round(disp_1[0], 4), round(disp_2[0], 4), round(disp_3[0], 4)]
        self.displacements = displacements

        return [F_12, F_21, F_23, F_32, displacements]

    def get_results(self):
        self.results()

        # Reactions
        self.R_1 = round(neg(self.P2[0][0]), 2)
        self.R_2 = round(neg(self.P2[1][0]), 2)
        self.R_3 = round(neg(self.P2[2][0]), 2)

        # F_12
        self.V_12 = round(self.F_12[0][0], 2)
        self.M_12 = round(self.F_12[1][0], 2)
        # F_21
        self.V_21 = round(self.F_21[0][0], 2)
        self.M_21 = round(self.F_21[1][0], 2)
        # F_23
        self.V_23 = round(self.F_23[0][0], 2)
        self.M_23 = round(self.F_23[1][0], 2)
        # F_32
        self.V_32 = round(self.F_32[0][0], 2)
        self.M_32 = round(self.F_32[1][0], 2)

        # Displacements
        self.slope_1 = round(self.displacements[0], 3)
        self.slope_2 = round(self.displacements[1], 3)
        self.slope_3 = round(self.displacements[2], 3)

        # Deflections # TODO: derive and add

        return [
            [self.R_1, self.R_2, self.R_3],
            [self.V_12, self.M_12],
            [self.V_21, self.M_21],
            [self.V_23, self.M_23],
            [self.V_32, self.M_32],
            [self.slope_1, self.slope_2, self.slope_3],
        ]

    def print_results(self):
        self.get_results()

        print(f"Reactions:")
        print(f"R_1: {self.R_1} kN")
        print(f"R_2: {self.R_2} kN")
        print(f"R_3: {self.R_3} kN")

        print(f"Internal forces:")
        print(f"V_12: {self.V_12} kN, M_12: {self.M_12} kNm")
        print(f"V_21: {self.V_21} kN, M_21: {self.M_21} kNm")
        print(f"V_23: {self.V_23} kN, M_23: {self.M_23} kNm")
        print(f"V_32: {self.V_32} kN, M_32: {self.M_32} kNm")

        print(f"Nodal Displacements:")
        print(f"Slope 1: {self.slope_1} radians")
        print(f"Slope 2: {self.slope_2} radians")
        print(f"Slope 3: {self.slope_3} radians")

    def get_reactions(self):
        self.get_results()
        return [self.R_1, self.R_2, self.R_3]

    def redistribution(self, percentage):
        """Returns redistributed moments at middle support and redistributed reactions"""

        self.results()

        # TODO: should recalc. reactions and shears from redistrubted moments in member-end forces
        re_factor = 1 - (
            percentage / 100
        )  # redistribution factor i.e if percentage = 15, re_factor = 0.85
        final_moments = [self.F_21[1], self.F_23[1]]

        lhs = final_moments[0] / self.span_12  # for readability
        rhs = final_moments[0] / self.span_23
        reactions_fx = [neg(lhs), (lhs + rhs), neg(rhs)]
        lhs_fr, rhs_fr = (self.udl_12 * self.span_12 / 2), (
            self.udl_23 * self.span_23 / 2
        )
        reactions_fr = [lhs_fr, (lhs_fr + rhs_fr), rhs_fr]
        reactions_fin = [
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

        self.reactions_fin = reactions_fin  # final reactions before redistribution
        self.reactions_fin_re = reactions_fin_re  # final reactions after redistribution

        return [self.re_moments, self.reactions_fin_re]


class MomentDistribution(BeamAnalysis):
    def __init__(self, spans, udls, EI_values=[1, 1]):
        super().__init__(spans, udls, EI_values)
        zeroes = np.zeros((13, 4))  # numpy array with 13 rows and 4 columns of zeroes
        columns = ["AB", "BA", "BC", "CB"]  # columns
        idxs = [
            "STIFF",
            "DF",
            "FEM",
            "BAL",
            "COM",
            "BAL",
            "COM",
            "BAL",
            "FMs",
            "ReFMs",
            "FxRx",
            "FrRx",
            "FnRx",
        ]  # rows
        self.df = pd.DataFrame(zeroes, columns=columns, index=idxs)

        pd.set_option("display.precision", 3)
        self.L_12 = spans[0]
        self.L_23 = spans[1]
        self.w_12 = udls[0]
        self.w_23 = udls[1]

        # self.re_f = 0.85 # FIXME: default value, should be user-defined

    def run(self, redistribution):
        # ->
        df = self.df
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
        df.iat[
            9, 0
        ] = 0  # Joint A # force set to 0 (for pinned end) to avoid floating pt errors
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

        self.R_1_dist = round2(df.iat[12, 0])  # joint A
        self.R_2_dist = round2(df.iat[12, 1] + df.iat[12, 2])  # joint B
        self.R_3_dist = round2(df.iat[12, 3])  # joint C

        self.df_final = df

        # --- Calc reactions pre-redist ---
        # -- Fixed reactions, pre-redist ---
        R_1fx = neg(df.iat[8, 1] / self.L_12)  # joint A
        R_2fx = neg(R_1fx) + neg(df.iat[8, 2] / self.L_23)  # joint B
        R_3fx = df.iat[8, 2] / self.L_23  # joint C

        # -- Final reactions, pre-redist. (Free reactions already calc.d)
        self.R_1_nodist = round2(R_1fx + df.iat[11, 0])
        self.R_2_nodist = round2(R_2fx + df.iat[11, 1] + df.iat[11, 2])  #
        self.R_3_nodist = round2(R_3fx + df.iat[11, 3])

        # return df
        # + if-elif-else for runs/errors & exceptions for diff't return types #TODO: try this
        self.md_df = f"""
|          | AB                      | BA                      | BC                      | CB                      |
| -------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| STIFF    | {round2(df.iat[0,0])}   | {round2(df.iat[0,1])}   | {round2(df.iat[0,2])}   | {round2(df.iat[0,3])}   |
| DF       | {round2(df.iat[1,0])}   | {round2(df.iat[1,1])}   | {round2(df.iat[1,2])}   | {round2(df.iat[1,3])}   |
| FEM      | {round2(df.iat[2,0])}   | {round2(df.iat[2,1])}   | {round2(df.iat[2,2])}   | {round2(df.iat[2,3])}   |
| BAL      | {round2(df.iat[3,0])}   | {round2(df.iat[3,1])}   | {round2(df.iat[3,2])}   | {round2(df.iat[3,3])}   |
| COM      | {round2(df.iat[4,0])}   | {round2(df.iat[4,1])}   | {round2(df.iat[4,2])}   | {round2(df.iat[4,3])}   |
| BAL      | {round2(df.iat[5,0])}   | {round2(df.iat[5,1])}   | {round2(df.iat[5,2])}   | {round2(df.iat[5,3])}   |
| COM      | {round2(df.iat[6,0])}   | {round2(df.iat[6,1])}   | {round2(df.iat[6,2])}   | {round2(df.iat[6,3])}   |
| BAL      | {round2(df.iat[7,0])}   | {round2(df.iat[7,1])}   | {round2(df.iat[7,2])}   | {round2(df.iat[7,3])}   |
| FMs      | {round2(df.iat[8,0])}   | {round2(df.iat[8,1])}   | {round2(df.iat[8,2])}   | {round2(df.iat[8,3])}   |
| ReFMs    | {round2(df.iat[9,0])}   | {round2(df.iat[9,1])}   | {round2(df.iat[9,2])}   | {round2(df.iat[9,3])}   |
| FxRx     | {round2(df.iat[10,0])}  | {round2(df.iat[10,1])}  | {round2(df.iat[10,2])}  | {round2(df.iat[10,3])}  |
| FrRx     | {round2(df.iat[11,0])}  | {round2(df.iat[11,1])}  | {round2(df.iat[11,2])}  | {round2(df.iat[11,3])}  |
| FnRx     | {round2(df.iat[12,0])}  | {round2(df.iat[12,1])}  | {round2(df.iat[12,2])}  | {round2(df.iat[12,3])}  |
"""

    # def get_results(self):
    #     return super().get_results() #TODO: FIXME:

    def get_results(self, redistribution):
        self.run(redistribution)
        """Returns a nested list of:
        \n`[moments before redist.]`,
        \n`[redistributed moments]`,
        \n`[reactions before redistr.]` and
        \n`[reactions after redistr.]`"""
        moments_pre_dist = [
            self.M_12_no_dist,
            self.M_21_no_dist,
            self.M_23_no_dist,
            self.M_32_no_dist,
        ]  #
        moments_redist = [
            self.M_12_dist,
            self.M_21_dist,
            self.M_23_dist,
            self.M_32_dist,
        ]
        reactions_pre_dist = [self.R_1_nodist, self.R_2_nodist, self.R_3_nodist]
        reactions_redist = [self.R_1_dist, self.R_2_dist, self.R_3_dist]

        return [moments_pre_dist, moments_redist, reactions_pre_dist, reactions_redist]

    def print_results(self, redistr_pct):
        self.run(redistr_pct)
        print(
            f"Moments (pre-redistr.): \n{[self.M_12_no_dist,self.M_21_no_dist,self.M_23_no_dist,self.M_32_no_dist]}"
        )
        print(
            f"Moments (redistr.): \n{[self.M_12_dist,self.M_21_dist,self.M_23_dist,self.M_32_dist]}"
        )
        print(
            f"Reactions (pre-redistr.): \n{[self.R_1_nodist,self.R_2_nodist,self.R_3_nodist]}"
        )
        print(f"Reactions (redistr.): \n{[self.R_1_dist,self.R_2_dist,self.R_3_dist]}")

    def get_table(self):
        return self.df_final

    def get_md_table(self):
        return self.md_df
    
    def get_results_st(self, redistr_pct):
        self.run(redistr_pct)
        results_str = f"Moments (pre-redistr.):  \n{[self.M_12_no_dist,self.M_21_no_dist,self.M_23_no_dist,self.M_32_no_dist]}  \nMoments (redistr.):  \n{[self.M_12_dist,self.M_21_dist,self.M_23_dist,self.M_32_dist]}  \nReactions (pre-redistr.):  \n{[self.R_1_nodist,self.R_2_nodist,self.R_3_nodist]}  \nReactions (redistr.):  \n{[self.R_1_dist,self.R_2_dist,self.R_3_dist]}"
        return results_str

    

