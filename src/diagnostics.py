import torch
import numpy as np


CP = 4.186


# =====================================================
# CALCULATE THERMAL RESIDUAL
# =====================================================

def calculate_thermal_residual(

    original,
    reconstructed,

    flow_idx,
    steam_temp_idx,
    feedwater_temp_idx
):

    original_energy = (

        original[:, :, flow_idx]

        * CP *

        (
            original[:, :, steam_temp_idx]
            -
            original[:, :, feedwater_temp_idx]
        )
    )

    reconstructed_energy = (

        reconstructed[:, :, flow_idx]

        * CP *

        (
            reconstructed[:, :, steam_temp_idx]
            -
            reconstructed[:, :, feedwater_temp_idx]
        )
    )

    residual = torch.mean(

        (
            original_energy -
            reconstructed_energy
        ) ** 2

    ).item()

    return residual


# =====================================================
# CALCULATE PRESSURE-FLOW RESIDUAL
# =====================================================

def calculate_pressure_residual(

    original,
    reconstructed,

    flow_idx,
    pressure_idx
):

    original_relation = (

        original[:, :, pressure_idx]
        *
        original[:, :, flow_idx]
    )

    reconstructed_relation = (

        reconstructed[:, :, pressure_idx]
        *
        reconstructed[:, :, flow_idx]
    )

    residual = torch.mean(

        (
            original_relation -
            reconstructed_relation
        ) ** 2

    ).item()

    return residual


# =====================================================
# GENERATE ENGINEERING EXPLANATION
# =====================================================

def generate_explanation(

    reconstruction_error,

    thermal_residual,
    pressure_residual
):

    explanations = []


    # ---------------------------------------------
    # Thermal anomaly
    # ---------------------------------------------

    if thermal_residual > 0.15:

        explanations.append(

            "Abnormal thermal-energy transfer detected "
            "between feedwater and steam system."
        )


    # ---------------------------------------------
    # Pressure-flow anomaly
    # ---------------------------------------------

    if pressure_residual > 0.08:

        explanations.append(

            "Steam-flow variation observed without "
            "expected pressure response."
        )


    # ---------------------------------------------
    # Combined instability
    # ---------------------------------------------

    if (
        thermal_residual > 0.15
        and
        pressure_residual > 0.08
    ):

        explanations.append(

            "Possible transient operating instability."
        )


    # ---------------------------------------------
    # Reconstruction anomaly
    # ---------------------------------------------

    if reconstruction_error > 0.18:

        explanations.append(

            "Sequence behavior deviates from "
            "learned normal operating patterns."
        )


    # ---------------------------------------------
    # Normal condition
    # ---------------------------------------------

    if len(explanations) == 0:

        explanations.append(
            "System operating normally."
        )


    return explanations