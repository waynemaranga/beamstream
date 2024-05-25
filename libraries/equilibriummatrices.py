"""Equilibrium Matrices"""

# Equilibrium matrices
from typing import Any
import numpy as np
from numpy import ndarray, dtype

# --- Span 1-2
beta_12: ndarray[Any, dtype[Any]] = np.array(object=[[-1.0, 0.0], [0.0, 1.0]])
beta_21: ndarray[Any, dtype[Any]] = np.array(object=[[1.0, 0.0], [0.0, 1.0]])

# --- Span 2-3
beta_23: ndarray[Any, dtype[Any]] = np.array(object=[[-1.0, 0.0], [0.0, 1.0]])
beta_32: ndarray[Any, dtype[Any]] = np.array(object=[[1.0, 0.0], [0.0, 1.0]])

# --- transposes
beta_12_T: ndarray[Any, dtype[Any]] = beta_12.transpose()
beta_21_T: ndarray[Any, dtype[Any]] = beta_21.transpose()
beta_23_T: ndarray[Any, dtype[Any]] = beta_23.transpose()
beta_32_T: ndarray[Any, dtype[Any]] = beta_32.transpose()
