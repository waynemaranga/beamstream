"""Beta Table, from Eurocode"""

from typing import Any

slab_types: list[str] = [
    "Interior panels",
    "One short edge discontinuous",
    "One long edge discontinuous",
    "Two adjacent edges discontinuous",
    "Two short edges discontinuous",
    "Two long edges discontinuous",
    "Three edges discontinuous (one long edge continuous)",
    "Three edges discontinuous (one short edge continuous)",
    "Four edges discontinuous",
]

type_a, type_b, type_c, type_d, type_e, type_f, type_g, type_h, type_i = slab_types

slab_type_list: list[str] = [
    type_a,
    type_b,
    type_c,
    type_d,
    type_e,
    type_f,
    type_g,
    type_h,
    type_i,
]

bound_list: list[float] = [
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.75,
    2.0,
]

slab_def_map: dict[str, str] = {
    "A": slab_type_list[0],
    "B": slab_type_list[1],
    "C": slab_type_list[2],
    "D": slab_type_list[3],
    "E": slab_type_list[4],
    "F": slab_type_list[5],
    "G": slab_type_list[6],
    "H": slab_type_list[7],
    "I": slab_type_list[8],
}


beta: dict[str, dict[str, Any]] = {
    type_a: {
        "type": type_a,
        "1.0": [0.031, 0.024],
        "1.1": [0.037, 0.028],
        "1.2": [0.042, 0.032],
        "1.3": [0.046, 0.035],
        "1.4": [0.050, 0.037],
        "1.5": [0.053, 0.040],
        "1.75": [0.059, 0.044],
        "2.0": [0.063, 0.048],
        "long": [0.032, 0.024],
    },
    type_b: {
        "type": type_b,
        "1.0": [0.039, 0.029],
        "1.1": [0.044, 0.033],
        "1.2": [0.048, 0.036],
        "1.3": [0.052, 0.039],
        "1.4": [0.055, 0.041],
        "1.5": [0.058, 0.043],
        "1.75": [0.063, 0.047],
        "2.0": [0.067, 0.050],
        "long": [0.037, 0.028],
    },
    type_c: {
        "type": type_c,
        "1.0": [0.039, 0.030],
        "1.1": [0.049, 0.036],
        "1.2": [0.056, 0.042],
        "1.3": [0.062, 0.047],
        "1.4": [0.068, 0.051],
        "1.5": [0.073, 0.055],
        "1.75": [0.082, 0.062],
        "2.0": [0.089, 0.067],
        "long": [0.037, 0.028],
    },
    type_d: {
        "type": type_d,
        "1.0": [0.047, 0.036],
        "1.1": [0.056, 0.042],
        "1.2": [0.063, 0.047],
        "1.3": [0.069, 0.051],
        "1.4": [0.074, 0.055],
        "1.5": [0.078, 0.059],
        "1.75": [0.087, 0.065],
        "2.0": [0.093, 0.070],
        "long": [0.045, 0.034],
    },
    type_e: {
        "type": type_e,
        "1.0": [0.046, 0.034],
        "1.1": [0.050, 0.038],
        "1.2": [0.054, 0.040],
        "1.3": [0.057, 0.043],
        "1.4": [0.060, 0.045],
        "1.5": [0.062, 0.047],
        "1.75": [0.067, 0.050],
        "2.0": [0.070, 0.053],
        "long": [0.0, 0.034],
    },
    type_f: {
        "type": type_f,
        "1.0": [0.000, 0.034],
        "1.1": [0.000, 0.046],
        "1.2": [0.000, 0.056],
        "1.3": [0.000, 0.065],
        "1.4": [0.000, 0.072],
        "1.5": [0.000, 0.078],
        "1.75": [0.000, 0.091],
        "2.0": [0.000, 0.100],
        "long": [0.045, 0.034],
    },
    type_g: {
        "type": type_g,
        "1.0": [0.057, 0.043],
        "1.1": [0.065, 0.048],
        "1.2": [0.071, 0.053],
        "1.3": [0.076, 0.057],
        "1.4": [0.081, 0.060],
        "1.5": [0.084, 0.063],
        "1.75": [0.092, 0.069],
        "2.0": [0.098, 0.074],
        "long": [0.000, 0.044],
    },
    type_h: {
        "type": type_h,
        "1.0": [0.000, 0.042],
        "1.1": [0.000, 0.054],
        "1.2": [0.000, 0.063],
        "1.3": [0.000, 0.071],
        "1.4": [0.000, 0.078],
        "1.5": [0.000, 0.084],
        "1.75": [0.000, 0.000],
        "2.0": [0.000, 0.105],
        "long": [0.058, 0.044],
    },
    type_i: {
        "type": type_i,
        "1.0": [0.000, 0.055],
        "1.1": [0.000, 0.065],
        "1.2": [0.000, 0.074],
        "1.3": [0.000, 0.081],
        "1.4": [0.000, 0.087],
        "1.5": [0.000, 0.092],
        "1.75": [0.000, 0.103],
        "2.0": [0.000, 0.111],
        "long": [0.000, 0.056],
    },
}
