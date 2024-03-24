# Equilibrium matrices
import numpy as np

# --- Span 1-2
beta_12 = np.array([[-1.0, 0.0], [0.0, 1.0]])
beta_21 = np.array([[1.0, 0.0], [0.0, 1.0]])

# --- Span 2-3
beta_23 = np.array([[-1.0, 0.0], [0.0, 1.0]])
beta_32 = np.array([[1.0, 0.0], [0.0, 1.0]])

# --- transposes
beta_12_T = beta_12.transpose()
beta_21_T = beta_21.transpose()
beta_23_T = beta_23.transpose()
beta_32_T = beta_32.transpose()
