# %%
from utilities.beamanalysis import BeamAnalysis, MomentDistribution
from utilities.plots import Plots
# from utilities.sheetspreader import Reader, Writer
from utilities.envelopes import ShearForceEnvelope, BendingMomentEnvelope

# %% [markdown]
# # Beam 2-2

# %% [markdown]
# ## Initialisation

# %%
# init
L_12 = 3.5 # m
L_23 = 3.5 

# %%
# -> Load case A: FULL-FULL
w_12A = 19.15 # TODO: beam loads should come from load transfers in slabs
w_23A = 21.54


# -> Load case B: 1-2 FULL, 2-3 EMPTY
w_12B = 10.55 # entering the init values for each load case should update the spreadsheet, or read from the spreadsheets
w_23B = 21.54

# -> Load case C: 1-2 EMPTY, 2-3 FULL
w_12C = 19.15 # entering the init values for each load case should update the spreadsheet, or read from the spreadsheets
w_23C = 11.77


# %%
# -> Load case A (FULL-FULL)
case_A = BeamAnalysis([L_12, L_23], [w_12A, w_23A])
case_A_rx = case_A.get_reactions()
case_A_plots = Plots([L_12, L_23],[w_12A,w_23A],case_A_rx)
case_A_re = case_A.redistribution(percentage=15)
case_A_re_rx = case_A.reactions_fin_re
case_A_plots_re = Plots([L_12, L_23],[w_12A,w_23A],case_A.reactions_fin_re)

# -> Load case B (1-2 FULL, 2-3 EMPTY)
case_B = BeamAnalysis([L_12, L_23], [w_12B, w_23B])
case_B_rx = case_B.get_reactions()
case_B_plots = Plots([L_12, L_23],[w_12B,w_23B],case_B_rx)
case_B_re = case_B.redistribution(percentage=15)
case_B_re_rx = case_B.reactions_fin_re
case_B_plots_re = Plots([L_12, L_23],[w_12B,w_23B],case_B.reactions_fin_re)

# -> Load case C (1-2 EMPTY, 2-3 FULL)
case_C = BeamAnalysis([L_12, L_23], [w_12C, w_23C])
case_C_rx = case_C.get_reactions()
case_C_plots = Plots([L_12, L_23],[w_12C,w_23C],case_C_rx)
case_C_re = case_C.redistribution(percentage=15)
case_C_re_rx = case_C.reactions_fin_re
case_C_plots_re = Plots([L_12, L_23],[w_12C,w_23C],case_C.reactions_fin_re)


# %% [markdown]
# ## Moment Distribution Method

# %% [markdown]
# ### Case_A

# %% [markdown]
# ### Case_B

# %%
case_B_mdm = MomentDistribution([L_12,L_23],[w_12B,w_23B])
case_B_mdm.run(15)
# print(case_B_mdm.get_results(15))
print(f"{case_B_mdm.get_table()}\n")
case_B_mdm.print_results(15)

# %% [markdown]
# ### Case_C

# %%
case_C_mdm = MomentDistribution([L_12,L_23],[w_12C,w_23C])
case_C_mdm.run(15)
# print(case_C_mdm.get_results(15))
print(f"{case_C_mdm.get_table()}\n")
# print(case_C_mdm.get_md_table())
case_C_mdm.print_results(15)

# %% [markdown]
# ## Envelopes

# %% [markdown]
# ### Shear Force Envelope

# %%
shear_envelope =  ShearForceEnvelope(case_A_plots_re, case_B_plots_re, case_C_plots_re)
shear_envelope.matplotlib()
shear_envelope.plotly()
# TODO: CHECK PLOT STRINGS AND PRINT THE BIGGEST OR SMALLEST VALUES
# KEY: RUN THE PLOT_STRINGS() METHOD ON THE PLOTS OBJECTS
# TODO: INSERT ANNOTATIONS FOR CRITICAL VALUES, MAX VALUES & MINIMUM 

# %% [markdown]
# ### Bending Moment Envelope

# %%
bending_envelope = BendingMomentEnvelope(case_A_plots_re, case_B_plots_re, case_C_plots_re)
bending_envelope.matplotlib()
bending_envelope.plotly()

# %% [markdown]
# 

# %% [markdown]
# 


