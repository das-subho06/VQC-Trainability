"""
Barren Plateau Analysis
"""

import numpy as np

from src.circuit import initialize_weights
from src.trainer import compute_gradient_norm
from src.config import NUM_RANDOM_INITIALIZATIONS


def analyze_barren_plateau(
    circuit,
    n_qubits,
    depth,
    X_train,
    y_train
):
    """
    Computes gradient statistics using multiple
    random parameter initializations.
    """

    gradient_norms = []

    print(f"\nAnalyzing {n_qubits} qubits | Depth {depth}")

    for run in range(NUM_RANDOM_INITIALIZATIONS):

        # Random initialization
        weights = initialize_weights(
            depth,
            n_qubits
        )

        # Gradient norm
        gradient = compute_gradient_norm(
            circuit,
            weights,
            X_train,
            y_train
        )

        gradient_norms.append(float(gradient))

        print(
            f"Run {run+1:03d} | "
            f"Gradient = {gradient:.8f}"
        )

    results = {

        "Qubits": n_qubits,

        "Depth": depth,

        "Mean Gradient":
            float(np.mean(gradient_norms)),

        "Std Gradient":
            float(np.std(gradient_norms)),

        "Minimum Gradient":
            float(np.min(gradient_norms)),

        "Maximum Gradient":
            float(np.max(gradient_norms)),

        "Gradient Samples":
            gradient_norms
    }

    return results