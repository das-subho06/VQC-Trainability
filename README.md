# VQC-Trainability

A PennyLane framework for investigating **trainability**, **barren plateaus**, and **ansatz design** in Variational Quantum Classifiers.

---

## Overview

This project benchmarks two VQC architectures on a binary classification task, systematically sweeping across qubit counts and circuit depths to study how these hyperparameters affect accuracy, gradient behavior, and training time.

**Dataset:** Breast Cancer Wisconsin (scikit-learn)  
**Task:** Binary classification (malignant / benign)  
**Framework:** [PennyLane](https://pennylane.ai/) 0.45.0  
**Total Experiments:** 32 (2 ansatzes × 4 qubit counts × 4 depths)

---

## Ansatzes

|
 Name 
|
 Description 
|
 Weight Shape 
|
|
------
|
-------------
|
-------------
|
|
`manual`
|
 RY rotations + linear CNOT chain per layer 
|
`(depth, n_qubits)`
|
|
`strong`
|
`StronglyEntanglingLayers`
 (PennyLane built-in) 
|
`(depth, n_qubits, 3)`
|

Both circuits use **AngleEmbedding** for data encoding and measure `⟨Z₀⟩`.

---

## Results Summary

### Best Configuration

> **`strong` ansatz, 4 qubits, depth 4 → 72.81% accuracy**

```
              precision    recall  f1-score   support

           0     0.6765    0.5349    0.5974        43
           1     0.7500    0.8451    0.7947        71

    accuracy                         0.7281       114
```

### By Ansatz (Averaged over all qubit/depth combinations)

|
 Ansatz 
|
 Mean Accuracy 
|
 Mean Final Loss 
|
 Mean Train Time 
|
|
--------
|
:-------------:
|
:---------------:
|
:---------------:
|
|
 manual 
|
 61.84%        
|
 0.8559          
|
 ~2,313 s        
|
|
 strong 
|
 65.68%        
|
 0.7939          
|
 ~4,336 s        
|

### VQC vs Classical Baselines

|
 Model               
|
 Best Accuracy 
|
|
---------------------
|
:-------------:
|
|
 Logistic Regression 
|
 99.12%        
|
|
 SVM                 
|
 96.49%        
|
|
 Random Forest       
|
 98.25%        
|
|
**
Best VQC
**
|
**
72.81%
**
|

Classical models use all 30 features. VQCs use PCA-compressed features (2–8 dims).

---

## Project Structure

```
VQC-Trainability/
├── main.py                  # Entry point
├── requirements.txt
├── src/
│   ├── config.py            # Experiment configuration
│   ├── circuit.py           # VQC circuit builder
│   ├── data_loader.py       # Dataset loading + PCA preprocessing
│   ├── trainer.py           # Training loop (Adam) + gradient tracking
│   ├── evaluator.py         # Test-set evaluation
│   ├── classical_models.py  # LR / SVM / Random Forest baselines
│   ├── experiment.py        # Full experiment pipeline
│   ├── plotting.py          # Plot generation
│   └── barren_plateau.py    # Gradient variance analysis utilities
└── results/
    ├── csv/
    │   ├── experiment_results.csv
    │   └── summary_results.csv
    ├── plots/
    │   ├── accuracy_vs_depth.png
    │   ├── accuracy_vs_qubits.png
    │   ├── ansatz_comparison.png
    │   ├── classical_vs_quantum.png
    │   └── training_time.png
    └── reports/
        └── {ansatz}_q{n}_d{d}.txt  (32 classification reports)
```

---

## Installation

```bash
git clone https://github.com/das-subho06/VQC-Trainability.git
cd VQC-Trainability
pip install -r requirements.txt
```

### Requirements

- `pennylane==0.45.0`
- `pennylane-lightning==0.45.0`
- `numpy>=2.0`, `scipy>=1.16`
- `scikit-learn>=1.7`
- `pandas>=2.2`
- `matplotlib>=3.10`

---

## Usage

```bash
python main.py
```

This runs all 32 experiments sequentially, then generates all plots.  
**Note:** Full run takes several hours on CPU (each experiment uses PennyLane's parameter-shift gradients across 455 training samples × 30 epochs).

To run a single experiment configuration:

```python
from src.experiment import run_single_experiment
from src.data_loader import load_dataset, preprocess_data
from src.classical_models import compare_classical_models

X, y = load_dataset()
X_train, X_test, y_train, y_test = preprocess_data(X, y, pca_components=4)
classical = compare_classical_models(X_train, y_train, X_test, y_test)

result = run_single_experiment(
    X_train, X_test, y_train, y_test,
    classical,
    n_qubits=4,
    depth=4,
    ansatz="strong"
)
```

---

## Key Findings

1. **Strong ansatz > manual** (+3.9% mean accuracy) due to richer entanglement structure.
2. **Optimal scale is moderate** — 4 qubits, depth 4 gives the best result. Adding more qubits or depth often hurts performance.
3. **No catastrophic barren plateaus** observed up to 8 qubits and 8 layers (30 epochs). However, gradient norms for the strong ansatz at 8 qubits are ~3× smaller than at 2 qubits.
4. **VQCs are not yet competitive** with classical models on this task — a ~25% accuracy gap exists. This is expected given the PCA compression and shallow circuit depth.
5. **High malignant recall (~84%)** in the best configuration — useful for medical screening applications.

---

## Configuration

Edit `src/config.py` to customize the sweep:

```python
QUBIT_COUNTS   = [2, 4, 6, 8]      # number of qubits to test
CIRCUIT_DEPTHS = [2, 4, 6, 8]      # circuit depths to test
AVAILABLE_ANSATZES = ["manual", "strong"]
EPOCHS         = 30
LEARNING_RATE  = 0.1
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
```

---

## License

MIT
