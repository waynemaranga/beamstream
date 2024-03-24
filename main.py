import streamlit as st
import pandas as pd

# -> Modules
from utilities.beamanalysis import BeamAnalysis, MomentDistribution
from utilities.plots import Plots
from st_envelopes import ShearForceEnvelope, BendingMomentEnvelope
from st_inputs import input_spans, input_cases


st.set_page_config(layout="wide", page_title="BeamStream\u00A9", page_icon=":hammer_and_wrench:")

# -> Top Header & Info
st.title('👷 BEAMSTREAM\u00A9: Beam Analysis')
st.subheader('Generate Moment Distribution Tables and Shear Force & Bending Moment Envelopes for a 2-span continuous beam with UDLs for 3 Load Cases.')
st.write('Available for free for a limited time only.')


# # -> Sidebar
st.sidebar.title('Contents: \u00A9')
st.sidebar.subheader('📝 Input')
st.sidebar.subheader('📊 Results')
st.sidebar.subheader('📋 Moment Distribution Tables')
st.sidebar.subheader('📈 Shear Force Envelope')
st.sidebar.subheader('💳 Bending Moment Envelope')
st.sidebar.write('Made in Nairobi, Kenya 🇰🇪 \u00A9 2023')

# -> User Inputs
# Span lengths
L_12, L_23 = input_spans('Input Span lengths', 'lengths')
 
# --- Case A
w_12A, w_23A, re_fA = input_cases('🇦 Load Case A', 'CASE_A')
# --- Case B
w_12B, w_23B, re_fB = input_cases('🇧 Load Case B', 'CASE_B')
# --- Case C
w_12C, w_23C, re_fC = input_cases('🇨 Load Case C', 'CASE_C')

if st.button('Run ▶️'):
    # st.write('Running...')
    st.balloons()

    # -> Load case A
    case_A = BeamAnalysis([L_12, L_23], [w_12A, w_23A])
    case_A_rx = case_A.get_reactions()
    case_A_plots = Plots([L_12, L_23],[w_12A,w_23A],case_A_rx)
    case_A_re = case_A.redistribution(percentage=re_fA)
    case_A_re_rx = case_A.reactions_fin_re
    case_A_plots_re = Plots([L_12, L_23],[w_12A,w_23A],case_A.reactions_fin_re)
    case_A_mdm = MomentDistribution([L_12,L_23],[w_12A,w_23A])
    case_A_mdm.run(re_fA)

    # -> Load case B
    case_B = BeamAnalysis([L_12, L_23], [w_12B, w_23B])
    case_B_rx = case_B.get_reactions()
    case_B_plots = Plots([L_12, L_23],[w_12B,w_23B],case_B_rx)
    case_B_re = case_B.redistribution(percentage=re_fB)
    case_B_re_rx = case_B.reactions_fin_re
    case_B_plots_re = Plots([L_12, L_23],[w_12B,w_23B],case_B.reactions_fin_re)
    case_B_mdm = MomentDistribution([L_12,L_23],[w_12B,w_23B])
    case_B_mdm.run(re_fB)

    # -> Load case C
    case_C = BeamAnalysis([L_12, L_23], [w_12C, w_23C])
    case_C_rx = case_C.get_reactions()
    case_C_plots = Plots([L_12, L_23],[w_12C,w_23C],case_C_rx)
    case_C_re = case_C.redistribution(percentage=re_fC)
    case_C_re_rx = case_C.reactions_fin_re
    case_C_plots_re = Plots([L_12, L_23],[w_12C,w_23C],case_C.reactions_fin_re)
    case_C_mdm = MomentDistribution([L_12,L_23],[w_12C,w_23C])
    case_C_mdm.run(re_fC)

    # --=> OUTPUTS
    # -> SECTION 1: Results & MDM Tables
    st.header('📐 Results')
    col_A, col_B, col_C = st.columns(3)
    with col_A:
        st.subheader('🇦 Case A')
        st.write(case_A_mdm.get_results_st(re_fA))  # Display textual results
    with col_B:
        st.subheader('🇧 Case B')
        st.write(case_B_mdm.get_results_st(re_fB))
    with col_C:
        st.subheader('🇨 Case C')
        st.write(case_C_mdm.get_results_st(re_fC))

    # -- Moment Distribution Method Tables
    st.header('📋 Moment Distribution Tables')
    
    st.caption('🇦 Case A')
    with st.expander("Case A: Click to Expand", expanded=False):
        st.dataframe(case_A_mdm.get_table())
    
    st.caption('🇧 Case B')
    with st.expander("Case B: Click to Expand", expanded=False):
        st.dataframe(case_B_mdm.get_table())

    st.caption('🇨 Case C')
    with st.expander("Case C: Click to Expand", expanded=False):
        st.dataframe(case_C_mdm.get_table())

    # -> SECTION 2: Envelopes
    st.header('📈 Envelopes')
    st.subheader('Shear Force Envelope')
    shear_force_envelope = ShearForceEnvelope(case_A_plots_re, case_B_plots_re, case_C_plots_re)
    # st.pyplot(shear_force_envelope.matplotlib()) # pyplot() renders a Matplotlib chart, needs a Matplotlib figure object as input
    st.plotly_chart(shear_force_envelope.plotly()) # plotly_chart() renders a Plotly chart, needs a Plotly figure object as input
    st.write('This is an interactive plot. Use your mouse to view values at different points, pan, zoom & save')


    st.subheader('Bending Moment Envelope')
    bending_moment_envelope = BendingMomentEnvelope(case_A_plots_re, case_B_plots_re, case_C_plots_re)
    st.plotly_chart(bending_moment_envelope.plotly())
    # st.pyplot(bending_moment_envelope.matplotlib())