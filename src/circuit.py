"""
Quantum Circuit Definition
"""

import pennylane as qml
from pennylane import numpy as np


def create_device(n_qubits):
    """
    Create a PennyLane quantum device.
    """
    return qml.device(
        "default.qubit",
        wires=n_qubits
    )


def build_circuit(device, n_qubits, depth, ansatz="manual"):
    """
    Creates a Variational Quantum Circuit.

    Parameters
    ----------
    device : PennyLane device
    n_qubits : int
    depth : int
    ansatz : str
        "manual"  -> RY + CNOT
        "strong"  -> StronglyEntanglingLayers
    """

    @qml.qnode(device)
    def circuit(features, weights):

        # -----------------------------
        # Data Encoding
        # -----------------------------
        qml.AngleEmbedding(
            features,
            wires=range(n_qubits)
        )

        # =====================================================
        # Manual Ansatz
        # =====================================================

        if ansatz == "manual":

            for d in range(depth):

                # Trainable Rotations
                for qubit in range(n_qubits):

                    qml.RY(
                        weights[d, qubit],
                        wires=qubit
                    )

                # Linear Entanglement
                for qubit in range(n_qubits - 1):

                    qml.CNOT(
                        wires=[qubit, qubit + 1]
                    )

        # =====================================================
        # Strongly Entangling Layers
        # =====================================================

        elif ansatz == "strong":

            qml.StronglyEntanglingLayers(
                weights,
                wires=range(n_qubits)
            )

        else:

            raise ValueError(
                f"Unknown ansatz: {ansatz}"
            )

        # -----------------------------
        # Measurement
        # -----------------------------
        return qml.expval(
            qml.PauliZ(0)
        )

    return circuit


def initialize_weights(
    depth,
    n_qubits,
    ansatz="manual"
):
    """
    Initialize trainable parameters.
    """

    if ansatz == "manual":

        return np.random.randn(
            depth,
            n_qubits,
            requires_grad=True
        )

    elif ansatz == "strong":

        return np.random.randn(
            depth,
            n_qubits,
            3,
            requires_grad=True
        )

    else:

        raise ValueError(
            f"Unknown ansatz: {ansatz}"
        )


def draw_circuit(
    circuit,
    sample,
    weights
):

    print(
        qml.draw(
            circuit,
            decimals=2
        )(sample, weights)
    )