import streamlit as st

def input_spans(subheader, key_suffix):
    st.subheader(subheader)
    # column definitions
    col_L12, col_L23 = st.columns(2) # don't use beta_columns
    # column contents
    with col_L12:
        L_12 = st.number_input('📏 Length of LHS span (1-2) [m]', key=f'L_12{key_suffix}', value=4) # unique key is needed for each input
    with col_L23:
        L_23 = st.number_input('📏 Length of RHS span (2-3) [m]', key=f'L_23{key_suffix}', value=5)
    return L_12, L_23


def input_cases(subheader, key_suffix):
    st.subheader(subheader)
    # column definitions
    col_w12, col_w23, col_re_f = st.columns(3) # don't use beta_columns
    # column contents
    with col_w12:
        w_12 = st.number_input('⏬⏬ UDL Span 1-2 [kN/m]', key=f'w_12{key_suffix}', value=20)
    with col_w23:
        w_23 = st.number_input('⏬⏬ UDL Span 2-3 [kN/m]', key=f'w_23{key_suffix}', value=24)
    with col_re_f:
        re_f = st.number_input('Redistribution Factor (%)', key=f're_f{key_suffix}', value=15)
    return w_12, w_23, re_f