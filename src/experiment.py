"""
Experiment Pipeline

Runs experiments for different:
- Ansatz
- Number of Qubits
- Circuit Depth
"""

import os
import pandas as pd

from src.config import (
    QUBIT_COUNTS,
    CIRCUIT_DEPTHS,
    AVAILABLE_ANSATZES
)

from src.data_loader import (
    load_dataset,
    preprocess_data
)

from src.circuit import (
    create_device,
    build_circuit,
    initialize_weights
)

from src.trainer import train
from src.evaluator import evaluate
from src.classical_models import compare_classical_models


def run_single_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    classical_results,
    n_qubits,
    depth,
    ansatz
):
    """
    Runs one VQC experiment.
    """

    # # ==========================================================
    # # Dataset Preprocessing
    # # ==========================================================

    # X_train, X_test, y_train, y_test = preprocess_data(
    #     X,
    #     y,
    #     pca_components=n_qubits
    # )

    # ==========================================================
    # Classical ML Baselines
    # ==========================================================

    # classical_results = compare_classical_models(
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test
    # )

    # ==========================================================
    # Quantum Device
    # ==========================================================

    device = create_device(
        n_qubits
    )

    # ==========================================================
    # Quantum Circuit
    # ==========================================================

    circuit = build_circuit(
        device=device,
        n_qubits=n_qubits,
        depth=depth,
        ansatz=ansatz
    )

    # ==========================================================
    # Initialize Weights
    # ==========================================================

    weights = initialize_weights(
        depth=depth,
        n_qubits=n_qubits,
        ansatz=ansatz
    )

    # ==========================================================
    # Train VQC
    # ==========================================================

    training_results = train(
        circuit=circuit,
        weights=weights,
        X_train=X_train,
        y_train=y_train
    )

    # ==========================================================
    # Evaluate VQC
    # ==========================================================

    accuracy, matrix, report, probabilities = evaluate(
        circuit=circuit,
        weights=training_results["weights"],
        X_test=X_test,
        y_test=y_test
    )
    os.makedirs(
        "results/reports",
        exist_ok=True
    )

    report_path = (
        f"results/reports/"
        f"{ansatz}_q{n_qubits}_d{depth}.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # ==========================================================
    # Return Results
    # ==========================================================

    return {

        "Ansatz": ansatz,

        "Qubits": n_qubits,

        "Depth": depth,

        # ---------------- Quantum ----------------

        "VQC Accuracy": accuracy,

        "Final Loss": training_results["final_loss"],

        "Average Gradient": training_results["average_gradient"],

        "Final Gradient": training_results["final_gradient"],

        "Training Time (s)": training_results["training_time"],

        # ---------------- Classical ----------------

        "Logistic Regression":
            classical_results["Logistic Regression"]["accuracy"],

        "Support Vector Machine":
            classical_results["Support Vector Machine"]["accuracy"],

        "Random Forest":
            classical_results["Random Forest"]["accuracy"]

        # "Classification Report":
        #     report,

        # "Confusion Matrix":
        #     matrix

    }


def run_all_experiments():
    """
    Runs every experiment configuration.
    """

    print("\nLoading Breast Cancer Dataset...\n")

    X, y = load_dataset()

    results = []

    os.makedirs(
        "results/reports",
        exist_ok=True
    )

    total_runs = (
        len(AVAILABLE_ANSATZES)
        * len(QUBIT_COUNTS)
        * len(CIRCUIT_DEPTHS)
    )

    current_run = 1



    for qubits in QUBIT_COUNTS:

        X_train, X_test, y_train, y_test = preprocess_data(
            X,
            y,
            pca_components=qubits
        )

        classical_results = compare_classical_models(
            X_train,
            y_train,
            X_test,
            y_test
        )
        for ansatz in AVAILABLE_ANSATZES:

            print("\n")
            print("=" * 80)
            print(f"ANSATZ : {ansatz.upper()}")
            print("=" * 80)

        # for qubits in QUBIT_COUNTS:

            for depth in CIRCUIT_DEPTHS:

                print("\n")
                print("-" * 80)
                print(
                    f"Experiment {current_run}/{total_runs}"
                )
                print(f"Ansatz : {ansatz}")
                print(f"Qubits : {qubits}")
                print(f"Depth  : {depth}")
                print("-" * 80)

                result = run_single_experiment(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    classical_results,
                    qubits,
                    depth,
                    ansatz
                )

                results.append(result)

                # with open(
                #     "results/reports/classification_report.txt",
                #     "a",
                #     encoding="utf-8"
                # ) as f:

                #     f.write("=" * 80 + "\n")

                #     f.write(f"Ansatz : {ansatz}\n")

                #     f.write(f"Qubits : {qubits}\n")

                #     f.write(f"Depth  : {depth}\n\n")

                #     f.write(result["Classification Report"])

                #     f.write("\n\n")

                print("\nResults")

                print(
                    f"VQC Accuracy              : {result['VQC Accuracy']:.4f}"
                )

                print(
                    f"Logistic Regression       : {result['Logistic Regression']:.4f}"
                )

                print(
                    f"Support Vector Machine    : {result['Support Vector Machine']:.4f}"
                )

                print(
                    f"Random Forest             : {result['Random Forest']:.4f}"
                )

                print(
                    f"Training Time (s)         : {result['Training Time (s)']:.2f}"
                )

                current_run += 1

    # ==========================================================
    # Save Results
    # ==========================================================

    df = pd.DataFrame(results)

    df = df.sort_values(
        by=[
            "Ansatz",
            "Qubits",
            "Depth"
        ]
    ).reset_index(drop=True)

    os.makedirs(
        "results/csv",
        exist_ok=True
    )

    csv_path = "results/csv/experiment_results.csv"

    df.to_csv(
        csv_path,
        index=False
    )

    summary = (
        df.groupby("Ansatz")
        .agg({

            "VQC Accuracy":"mean",

            "Final Loss":"mean",

            "Training Time (s)":"mean",

            "Average Gradient":"mean",

            "Final Gradient":"mean"

        })
        .reset_index()
    )

    summary.to_csv(
        "results/csv/summary_results.csv",
        index=False
    )

    print(summary)

    print("\n")
    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)

    print(df)

    print(f"\nResults saved to: {csv_path}")

    return df