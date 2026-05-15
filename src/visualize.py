import matplotlib.pyplot as plt
import numpy as np


# =====================================================
# PLOT RECONSTRUCTION ERRORS
# =====================================================

def plot_reconstruction_errors(errors, threshold):

    plt.figure(figsize=(14, 6))

    plt.plot(
        errors,
        label="Reconstruction Error"
    )

    plt.axhline(
        y=threshold,
        color='r',
        linestyle='--',
        label='Threshold'
    )

    plt.title(
        "Reconstruction Error Over Time"
    )

    plt.xlabel("Sequence Index")
    plt.ylabel("Error")

    plt.legend()

    plt.tight_layout()

    plt.show()


# =====================================================
# PLOT DETECTED ANOMALIES
# =====================================================

def plot_anomalies(errors, threshold):

    anomalies = errors > threshold

    anomaly_indices = np.where(
        anomalies
    )[0]

    plt.figure(figsize=(14, 6))

    plt.plot(
        errors,
        label="Reconstruction Error"
    )

    plt.scatter(
        anomaly_indices,
        errors[anomaly_indices],
        color='red',
        label='Anomalies'
    )

    plt.axhline(
        y=threshold,
        color='black',
        linestyle='--',
        label='Threshold'
    )

    plt.title("Detected Anomalies")

    plt.xlabel("Sequence Index")
    plt.ylabel("Error")

    plt.legend()

    plt.tight_layout()

    plt.show()


# =====================================================
# PLOT SIGNALS WITH ANOMALIES
# =====================================================

def plot_signal_with_anomalies(
    df,
    errors,
    threshold,
    column_name,
    window_size=20,
    start=0,
    end=2000
):

    anomalies = errors > threshold

    anomaly_indices = np.where(
        anomalies
    )[0]

    # Shift indices because sequences
    # start after window_size
    anomaly_indices = (
        anomaly_indices + window_size
    )

    # Keep only anomalies inside range
    mask = (
        (anomaly_indices >= start) &
        (anomaly_indices < end)
    )

    anomaly_indices = anomaly_indices[
        mask
    ]

    plt.figure(figsize=(14, 6))

    # Plot selected signal window
    plt.plot(
        df.index[start:end],
        df[column_name].iloc[start:end],
        label=column_name
    )

    # Plot anomaly points
    plt.scatter(
        df.index[anomaly_indices],
        df[column_name].iloc[anomaly_indices],
        color='red',
        label='Anomalies'
    )

    plt.title(
        f"{column_name} with Detected Anomalies"
    )

    plt.xlabel("Time")
    plt.ylabel(column_name)

    plt.legend()

    plt.tight_layout()

    plt.show()