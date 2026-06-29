"""
Model Evaluation
"""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def predict(circuit, x, weights):
    """
    Predict the class of a single sample.

    Parameters
    ----------
    circuit : QNode
    x : numpy array
    weights : trainable parameters

    Returns
    -------
    int
        0 or 1
    """

    expectation = circuit(x, weights)

    prediction = 1 if expectation >= 0 else 0

    return prediction


def predict_proba(circuit, x, weights):
    """
    Returns the raw expectation value.

    Used for analysis and future ROC-AUC computation.
    """

    return circuit(x, weights)


def evaluate(
    circuit,
    weights,
    X_test,
    y_test
):
    """
    Evaluate model on the test set.

    Returns
    -------
    accuracy
    confusion matrix
    classification report
    """

    predictions = []

    probabilities = []

    for sample in X_test:

        predictions.append(
            predict(
                circuit,
                sample,
                weights
            )
        )

        probabilities.append(
            predict_proba(
                circuit,
                sample,
                weights
            )
        )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        digits=4
    )

    return (
        accuracy,
        matrix,
        report,
        probabilities
    )