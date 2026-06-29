"""
Visualization Utilities
"""

import os

import matplotlib.pyplot as plt
import pandas as pd


def load_results(csv_path="results/csv/experiment_results.csv"):
    """
    Load experiment results.
    """

    return pd.read_csv(csv_path)


def create_plot_directory():

    os.makedirs(
        "results/plots",
        exist_ok=True
    )


# ==========================================================
# Accuracy vs Depth
# ==========================================================

def plot_accuracy_vs_depth(df):

    plt.figure(figsize=(8,5))

    for ansatz in df["Ansatz"].unique():

        subset = (
            df[df["Ansatz"] == ansatz]
            .groupby("Depth")["VQC Accuracy"]
            .mean()
        )

        plt.plot(
            subset.index,
            subset.values,
            marker="o",
            linewidth=2,
            label=ansatz
        )

    plt.title("Accuracy vs Circuit Depth")

    plt.xlabel("Circuit Depth")

    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/plots/accuracy_vs_depth.png"
    )

    plt.close()


# ==========================================================
# Accuracy vs Qubits
# ==========================================================

def plot_accuracy_vs_qubits(df):

    plt.figure(figsize=(8,5))

    for ansatz in df["Ansatz"].unique():

        subset = (
            df[df["Ansatz"] == ansatz]
            .groupby("Qubits")["VQC Accuracy"]
            .mean()
        )

        plt.plot(
            subset.index,
            subset.values,
            marker="o",
            linewidth=2,
            label=ansatz
        )

    plt.title("Accuracy vs Number of Qubits")

    plt.xlabel("Qubits")

    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/plots/accuracy_vs_qubits.png"
    )

    plt.close()


# ==========================================================
# Training Time
# ==========================================================

def plot_training_time(df):

    plt.figure(figsize=(8,5))

    for ansatz in df["Ansatz"].unique():

        subset = (
            df[df["Ansatz"] == ansatz]
            .groupby("Qubits")["Training Time (s)"]
            .mean()
        )

        plt.plot(
            subset.index,
            subset.values,
            marker="o",
            linewidth=2,
            label=ansatz
        )

    plt.title("Training Time vs Number of Qubits")

    plt.xlabel("Qubits")

    plt.ylabel("Training Time (seconds)")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/plots/training_time.png"
    )

    plt.close()


# ==========================================================
# Ansatz Comparison
# ==========================================================

def plot_ansatz_comparison(df):

    comparison = (
        df.groupby("Ansatz")["VQC Accuracy"]
        .mean()
    )

    plt.figure(figsize=(6,5))

    comparison.plot(kind="bar")

    plt.ylabel("Average Accuracy")

    plt.title("Average Accuracy of Quantum Ansatzes")

    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        "results/plots/ansatz_comparison.png"
    )

    plt.close()


# ==========================================================
# Classical vs Quantum
# ==========================================================

def plot_classical_vs_quantum(df):

    comparison = {

        "VQC":
            df["VQC Accuracy"].mean(),

        "Logistic":
            df["Logistic Regression"].mean(),

        "SVM":
            df["Support Vector Machine"].mean(),

        "Random Forest":
            df["Random Forest"].mean()

    }

    plt.figure(figsize=(7,5))

    plt.bar(
        comparison.keys(),
        comparison.values()
    )

    plt.ylabel("Average Accuracy")

    plt.title("Quantum vs Classical Models")

    plt.grid(axis="y")

    plt.tight_layout()

    plt.savefig(
        "results/plots/classical_vs_quantum.png"
    )

    plt.close()


# ==========================================================
# Generate All Plots
# ==========================================================

def generate_all_plots():

    create_plot_directory()

    df = load_results()

    plot_accuracy_vs_depth(df)

    plot_accuracy_vs_qubits(df)

    plot_training_time(df)

    plot_ansatz_comparison(df)

    plot_classical_vs_quantum(df)

    print("\nPlots generated successfully!")
