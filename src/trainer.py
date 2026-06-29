"""
Training Utilities
"""

import time

import pennylane as qml
from pennylane import numpy as np

from src.config import (
    EPOCHS,
    LEARNING_RATE,
    TRAIN_SAMPLES
)


def loss(circuit, weights, X, y):
    """
    Mean Squared Error Loss
    """

    losses = []

    for features, label in zip(X, y):

        target = 1 if label == 1 else -1

        prediction = circuit(
            features,
            weights
        )

        losses.append(
            (prediction - target) ** 2
        )

    return qml.math.mean(
        qml.math.stack(losses)
    )


def predict(circuit, x, weights):
    """
    Binary Prediction
    """

    prediction = circuit(
        x,
        weights
    )

    return 1 if prediction > 0 else 0


def compute_gradient_norm(
    circuit,
    weights,
    X,
    y
):
    """
    Computes ||gradient||
    """

    grad_fn = qml.grad(
        lambda w: loss(
            circuit,
            w,
            X,
            y
        )
    )

    gradients = grad_fn(weights)

    return np.linalg.norm(gradients)


def train(
    circuit,
    weights,
    X_train,
    y_train
):
    """
    Train the VQC
    """

    optimizer = qml.AdamOptimizer(
        stepsize=LEARNING_RATE
    )
    # Use either the full training set or a subset
    if TRAIN_SAMPLES is None:
        X_batch = X_train
        y_batch = y_train
    else:
        X_batch = X_train[:TRAIN_SAMPLES]
        y_batch = y_train[:TRAIN_SAMPLES]

    loss_history = []

    gradient_history = []

    start_time = time.time()

    for epoch in range(EPOCHS):

        weights = optimizer.step(

            lambda w: loss(
                circuit,
                w,
                X_batch,
                y_batch
            ),

            weights
        )

        current_loss = loss(
            circuit,
            weights,
            X_batch,
            y_batch
        )

        current_gradient = compute_gradient_norm(
            circuit,
            weights,
            X_batch,
            y_batch
        )

        loss_history.append(
            current_loss
        )

        gradient_history.append(
            current_gradient
        )

        print(
            f"Epoch {epoch+1:02d} | "
            f"Loss = {current_loss:.4f} | "
            f"Gradient Norm = {current_gradient:.6f}"
        )

    training_time = (
        time.time()
        -
        start_time
    )
    final_loss = float(loss_history[-1])

    average_gradient = float(
        np.mean(gradient_history)
    )

    final_gradient = float(
        gradient_history[-1]
    )
    return {
    "weights": weights,
    "loss_history": loss_history,
    "gradient_history": gradient_history,
    "training_time": training_time,
    "final_loss": final_loss,
    "average_gradient": average_gradient,
    "final_gradient": final_gradient
}