import torch


# =====================================================
# CONSTANTS
# =====================================================

CP = 4.186


# =====================================================
# THERMAL ENERGY CALCULATION
# =====================================================

def calculate_energy(
    sequence,
    flow_idx,
    steam_temp_idx,
    feedwater_temp_idx
):

    flow = sequence[:, :, flow_idx]

    steam_temp = sequence[:, :, steam_temp_idx]

    feedwater_temp = sequence[:, :, feedwater_temp_idx]

    energy = (
        flow *
        CP *
        (
            steam_temp -
            feedwater_temp
        )
    )

    return energy


# =====================================================
# THERMAL PHYSICS LOSS
# =====================================================

def thermal_physics_loss(
    original,
    reconstructed,
    flow_idx,
    steam_temp_idx,
    feedwater_temp_idx
):

    original_energy = calculate_energy(
        original,
        flow_idx,
        steam_temp_idx,
        feedwater_temp_idx
    )

    reconstructed_energy = calculate_energy(
        reconstructed,
        flow_idx,
        steam_temp_idx,
        feedwater_temp_idx
    )

    loss = torch.mean(
        (
            original_energy -
            reconstructed_energy
        ) ** 2
    )

    return loss


# =====================================================
# PRESSURE-FLOW PHYSICS LOSS
# =====================================================

def pressure_flow_loss(
    original,
    reconstructed,
    flow_idx,
    pressure_idx
):

    original_flow = original[:, :, flow_idx]

    original_pressure = original[:, :, pressure_idx]

    reconstructed_flow = (
        reconstructed[:, :, flow_idx]
    )

    reconstructed_pressure = (
        reconstructed[:, :, pressure_idx]
    )

    # Stable pressure-flow relationship
    original_relation = (
        original_pressure *
        original_flow
    )

    reconstructed_relation = (
        reconstructed_pressure *
        reconstructed_flow
    )

    loss = torch.mean(
        (
            original_relation -
            reconstructed_relation
        ) ** 2
    )

    return loss


# =====================================================
# TOTAL PHYSICS LOSS
# =====================================================

def total_physics_loss(
    original,
    reconstructed,
    flow_idx,
    steam_temp_idx,
    feedwater_temp_idx,
    pressure_idx
):

    thermal_loss = thermal_physics_loss(
        original,
        reconstructed,
        flow_idx,
        steam_temp_idx,
        feedwater_temp_idx
    )

    pressure_loss = pressure_flow_loss(
        original,
        reconstructed,
        flow_idx,
        pressure_idx
    )

    total_loss = (
        thermal_loss +
        pressure_loss
    )

    return (
        total_loss,
        thermal_loss,
        pressure_loss
    )