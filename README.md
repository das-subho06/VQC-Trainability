# VQC-Trainability

> A PennyLane-based framework for analyzing the **trainability** of Variational Quantum Classifiers (VQCs) by comparing different ansatz architectures, qubit counts, and circuit depths.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.45.0-purple)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project investigates how the architecture of a **Variational Quantum Classifier (VQC)** influences its trainability and predictive performance.

Experiments are performed on the **Breast Cancer Wisconsin Dataset** by varying:

- Number of qubits
- Circuit depth
- Quantum ansatz architecture

Two quantum circuit designs are benchmarked and compared against classical machine learning models.

### Dataset

| Property | Value |
|----------|-------|
| Dataset | Breast Cancer Wisconsin |
| Task | Binary Classification |
| Classes | Malignant / Benign |
| Framework | PennyLane 0.45.0 |
| Total Experiments | 32 |

---

## Quantum Ansatzes

| Ansatz | Description | Weight Shape |
|---------|-------------|--------------|
| **manual** | Custom RY rotations with Linear CNOT chain | `(depth, n_qubits)` |
| **strong** | PennyLane `StronglyEntanglingLayers` | `(depth, n_qubits, 3)` |

Both circuits use **AngleEmbedding** for feature encoding and measure the expectation value of **Pauli-Z** on the first qubit.

---

## Results

### Best Configuration

| Parameter | Value |
|-----------|-------|
| Ansatz | **Strongly Entangling Layers** |
| Qubits | **4** |
| Depth | **4** |
| Accuracy | **72.81%** |

### Classification Report

```text
              precision    recall  f1-score   support

           0     0.6765    0.5349    0.5974        43
           1     0.7500    0.8451    0.7947        71

    accuracy                         0.7281       114
```

### Average Performance

| Ansatz | Mean Accuracy | Mean Final Loss | Mean Training Time |
|---------|--------------:|----------------:|-------------------:|
| **Manual** | 61.84% | 0.8559 | ~2313 s |
| **Strong** | **65.68%** | **0.7939** | ~4336 s |

### Classical vs Quantum

| Model | Accuracy |
|-------|---------:|
|Best Logistic Regression | **99.12%** |
|Best Random Forest | 98.25% |
|Best Support Vector Machine | 96.37% |
| **Best VQC** | **72.81%** |

> **Note:** Classical models and VQCs are evaluated on the same PCA-compressed feature space, where the number of principal components matches the selected qubit count.

---

## Project Structure

```text
VQC-Trainability/
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── src/
│   ├── config.py
│   ├── circuit.py
│   ├── data_loader.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── classical_models.py
│   ├── experiment.py
│   ├── plotting.py
│   
│
└── results/
    ├── csv/
    ├── plots/
    └── reports/
```

---

## Installation

```bash
git clone https://github.com/das-subho06/VQC-Trainability.git

cd VQC-Trainability

pip install -r requirements.txt
```

---

## Requirements

- PennyLane 0.45.0
- PennyLane Lightning 0.45.0
- NumPy
- SciPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Running Experiments

Run the complete benchmark:

```bash
python main.py
```

The pipeline automatically:

- Loads the dataset
- Applies preprocessing (StandardScaler + PCA)
- Trains classical baselines
- Trains both VQC architectures
- Evaluates every configuration
- Saves experiment results
- Generates plots
- Stores classification reports

---

## Configuration

Modify `src/config.py` to customize the experiments.

```python
QUBIT_COUNTS = [2, 4, 6, 8]

CIRCUIT_DEPTHS = [2, 4, 6, 8]

AVAILABLE_ANSATZES = [
    "manual",
    "strong"
]

EPOCHS = 30

LEARNING_RATE = 0.1

TEST_SIZE = 0.20

RANDOM_STATE = 42
```

---

## Key Findings

- Strongly Entangling Layers consistently outperform the manually designed ansatz.
- Moderate circuit sizes (4 qubits, depth 4) achieve the best overall performance.
- Increasing qubits and circuit depth does not necessarily improve classification accuracy.
- Classical machine learning models outperform the investigated VQCs on this dataset.
- The framework enables systematic benchmarking of different VQC architectures and their training behaviour.

---

## Future Work

Potential improvements include:

- Mini-batch quantum training
- Alternative data embedding strategies
- Additional variational ansatz architectures
- Noise-aware quantum simulations
- Hybrid quantum-classical optimization

---

## License

This project is licensed under the **MIT License**.
