import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from src.load_data import load_and_prepare_data

from src.preprocess import (
    select_features,
    scale_features,
    create_sequences
)

from src.model import LSTMAutoencoder

from src.anomaly import (
    calculate_reconstruction_error
)

from src.visualize import (
    plot_reconstruction_errors,
    plot_anomalies,
    plot_signal_with_anomalies
)

from src.physics import (
    total_physics_loss
)


# =====================================================
# LOAD DATA
# =====================================================

df = load_and_prepare_data(
    "data/loop_dataset.csv"
)


# =====================================================
# FEATURE SELECTION
# =====================================================

df_selected = select_features(df)


# =====================================================
# SCALE FEATURES
# =====================================================

df_scaled, scaler = scale_features(
    df_selected
)


# =====================================================
# CREATE TIME SEQUENCES
# =====================================================

sequences = create_sequences(
    df_scaled.values,
    window_size=20
)


# =====================================================
# CONVERT TO TENSOR
# =====================================================

sequences = np.array(sequences)

sequences = torch.tensor(
    sequences,
    dtype=torch.float32
)

print("\nTensor Shape:")
print(sequences.shape)


# =====================================================
# FEATURE INDICES
# =====================================================

FLOW_IDX = 0
STEAM_TEMP_IDX = 1
PRESSURE_IDX = 2
FEEDWATER_FLOW_IDX = 3
FEEDWATER_TEMP_IDX = 4


# =====================================================
# INITIALIZE MODEL
# =====================================================

n_features = sequences.shape[2]

model = LSTMAutoencoder(
    n_features
)

print("\nModel Architecture:")
print(model)


# =====================================================
# LOSS FUNCTION & OPTIMIZER
# =====================================================

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# =====================================================
# TRAIN MODEL
# =====================================================

epochs = 10

print("\nStarting PINN Training...\n")

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    # ---------------------------------------------
    # Forward pass
    # ---------------------------------------------

    output = model(sequences)

    # ---------------------------------------------
    # Reconstruction loss
    # ---------------------------------------------

    reconstruction_loss = criterion(
        output,
        sequences
    )

    # ---------------------------------------------
    # Physics-informed loss
    # ---------------------------------------------

    phys_loss, thermal_loss, pressure_loss = (
        total_physics_loss(
            sequences,
            output,
            FLOW_IDX,
            STEAM_TEMP_IDX,
            FEEDWATER_TEMP_IDX,
            PRESSURE_IDX
        )
    )

    loss = (
        reconstruction_loss +
        0.035 * phys_loss
    )

    # ---------------------------------------------
    # Backpropagation
    # ---------------------------------------------

    loss.backward()

    optimizer.step()

    # ---------------------------------------------
    # Logging
    # ---------------------------------------------

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Total: {loss.item():.6f} | "
        f"Recon: {reconstruction_loss.item():.6f} | "
        f"Physics: {phys_loss.item():.6f} | "
        f"Thermal: {thermal_loss.item():.6f} | "
        f"Pressure: {pressure_loss.item():.6f}"
    )

print("\nPINN Training Complete!")


# =====================================================
# CALCULATE RECONSTRUCTION ERRORS
# =====================================================

errors = calculate_reconstruction_error(
    model,
    sequences
)

print("\nSample Reconstruction Errors:")
print(errors[:10])


# =====================================================
# DEFINE ANOMALY THRESHOLD
# =====================================================

threshold = np.percentile(
    errors,
    95
)

print(
    f"\nAnomaly Threshold: "
    f"{threshold:.6f}"
)


# =====================================================
# DETECT ANOMALIES
# =====================================================

anomalies = errors > threshold

print(
    f"\nDetected anomalies: "
    f"{anomalies.sum()}"
)


# =====================================================
# VISUALIZATIONS
# =====================================================

# Reconstruction error plot
plot_reconstruction_errors(
    errors,
    threshold
)

# ML anomaly plot
plot_anomalies(
    errors,
    threshold
)


# =====================================================
# SIGNAL VISUALIZATION WINDOW
# =====================================================

START = 0
END = 2000


# =====================================================
# STEAM FLOW ANOMALIES
# =====================================================

plot_signal_with_anomalies(
    df_selected,
    errors,
    threshold,
    "Main steam flow (t/h)",
    start=START,
    end=END
)


# =====================================================
# STEAM TEMPERATURE ANOMALIES
# =====================================================

plot_signal_with_anomalies(
    df_selected,
    errors,
    threshold,
    "Main steam temperature (boiler side) (℃)",
    start=START,
    end=END
)


# =====================================================
# STEAM PRESSURE ANOMALIES
# =====================================================

plot_signal_with_anomalies(
    df_selected,
    errors,
    threshold,
    "Main steam pressure (boiler side) (Mpa)",
    start=START,
    end=END
)