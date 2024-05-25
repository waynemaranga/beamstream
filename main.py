"""Main App"""

# pylint: disable = C0301


from typing import Any, Union
import streamlit as st

# import pandas as pd

# -> Modules
from utilities.beamanalysis import BeamAnalysis, MomentDistribution
from utilities.plots import Plots
from st_envelopes import ShearForceEnvelope, BendingMomentEnvelope
from st_inputs import input_spans, input_cases


st.set_page_config(
    layout="wide", page_title="BeamStream\u00A9", page_icon=":hammer_and_wrench:"
)

# -> Top Header & Info
st.title(body="👷 BEAMSTREAM\u00A9: Beam Analysis")
st.subheader(
    body="Generate Moment Distribution Tables and Shear Force & Bending Moment Envelopes for a 2-span continuous beam with UDLs for 3 Load Cases."
)
st.write("Available for free for a limited time only.")


# # -> Sidebar
st.sidebar.title(body="Contents: \u00A9")
st.sidebar.subheader(body="📝 Input")
st.sidebar.subheader(body="📊 Results")
st.sidebar.subheader(body="📋 Moment Distribution Tables")
st.sidebar.subheader(body="📈 Shear Force Envelope")
st.sidebar.subheader(body="💳 Bending Moment Envelope")
st.sidebar.write("Made in Nairobi, Kenya 🇰🇪 \u00A9 2023")

# -> User Inputs
# Span lengths
Number = Union[float, int]
L_12, L_23 = input_spans(subheader="Input Span lengths", key_suffix="lengths")

# --- Case A
w_12A, w_23A, re_fA = input_cases(subheader="🇦 Load Case A", key_suffix="CASE_A")
# --- Case B
w_12B, w_23B, re_fB = input_cases(subheader="🇧 Load Case B", key_suffix="CASE_B")
# --- Case C
w_12C, w_23C, re_fC = input_cases(subheader="🇨 Load Case C", key_suffix="CASE_C")

if st.button(label="Run ▶️"):
    # st.write('Running...')
    st.balloons()

    # -> Load case A
    case_A = BeamAnalysis(spans=[L_12, L_23], udls=[w_12A, w_23A])
    case_A_rx: list[Any] = case_A.get_reactions()
    case_A_plots = Plots(spans=[L_12, L_23], udls=[w_12A, w_23A], reactions=case_A_rx)
    case_A_re: list[list[Any]] = case_A.redistribution(percentage=re_fA)
    case_A_re_rx = case_A.reactions_fin_re
    case_A_plots_re = Plots(
        spans=[L_12, L_23], udls=[w_12A, w_23A], reactions=case_A.reactions_fin_re
    )
    case_A_mdm = MomentDistribution(spans=[L_12, L_23], udls=[w_12A, w_23A])
    case_A_mdm.run(redistribution=re_fA)

    # -> Load case B
    case_B = BeamAnalysis(spans=[L_12, L_23], udls=[w_12B, w_23B])
    case_B_rx: list[Any] = case_B.get_reactions()
    case_B_plots = Plots(spans=[L_12, L_23], udls=[w_12B, w_23B], reactions=case_B_rx)
    case_B_re: list[list[Any]] = case_B.redistribution(percentage=re_fB)
    case_B_re_rx = case_B.reactions_fin_re
    case_B_plots_re = Plots(
        spans=[L_12, L_23], udls=[w_12B, w_23B], reactions=case_B.reactions_fin_re
    )
    case_B_mdm = MomentDistribution(spans=[L_12, L_23], udls=[w_12B, w_23B])
    case_B_mdm.run(redistribution=re_fB)

    # -> Load case C
    case_C = BeamAnalysis(spans=[L_12, L_23], udls=[w_12C, w_23C])
    case_C_rx: list[Any] = case_C.get_reactions()
    case_C_plots = Plots(spans=[L_12, L_23], udls=[w_12C, w_23C], reactions=case_C_rx)
    case_C_re: list[list[Any]] = case_C.redistribution(percentage=re_fC)
    case_C_re_rx = case_C.reactions_fin_re
    case_C_plots_re = Plots(
        spans=[L_12, L_23], udls=[w_12C, w_23C], reactions=case_C.reactions_fin_re
    )
    case_C_mdm = MomentDistribution(spans=[L_12, L_23], udls=[w_12C, w_23C])
    case_C_mdm.run(redistribution=re_fC)

    # --=> OUTPUTS
    # -> SECTION 1: Results & MDM Tables
    st.header(body="📐 Results")
    col_A, col_B, col_C = st.columns(3)
    with col_A:
        st.subheader(body="🇦 Case A")
        st.write(
            case_A_mdm.get_results_st(redistr_pct=re_fA)
        )  # Display textual results
    with col_B:
        st.subheader(body="🇧 Case B")
        st.write(case_B_mdm.get_results_st(redistr_pct=re_fB))
    with col_C:
        st.subheader(body="🇨 Case C")
        st.write(case_C_mdm.get_results_st(redistr_pct=re_fC))

    # -- Moment Distribution Method Tables
    st.header(body="📋 Moment Distribution Tables")

    st.caption(body="🇦 Case A")
    with st.expander(label="Case A: Click to Expand", expanded=False):
        st.dataframe(data=case_A_mdm.get_table())

    st.caption(body="🇧 Case B")
    with st.expander(label="Case B: Click to Expand", expanded=False):
        st.dataframe(data=case_B_mdm.get_table())

    st.caption(body="🇨 Case C")
    with st.expander(label="Case C: Click to Expand", expanded=False):
        st.dataframe(data=case_C_mdm.get_table())

    # -> SECTION 2: Envelopes
    st.header(body="📈 Envelopes")
    st.subheader(body="Shear Force Envelope")
    shear_force_envelope = ShearForceEnvelope(
        case_1=case_A_plots_re, case_2=case_B_plots_re, case_3=case_C_plots_re
    )
    # st.pyplot(shear_force_envelope.matplotlib()) # pyplot() renders a Matplotlib chart, needs a Matplotlib figure object as input
    st.plotly_chart(
        figure_or_data=shear_force_envelope.plotly()
    )  # plotly_chart() renders a Plotly chart, needs a Plotly figure object as input
    st.write(
        "This is an interactive plot. Use your mouse to view values at different points, pan, zoom & save"
    )

    st.subheader(body="Bending Moment Envelope")
    bending_moment_envelope = BendingMomentEnvelope(
        case_1=case_A_plots_re, case_2=case_B_plots_re, case_3=case_C_plots_re
    )
    st.plotly_chart(figure_or_data=bending_moment_envelope.plotly())
    # st.pyplot(bending_moment_envelope.matplotlib())
