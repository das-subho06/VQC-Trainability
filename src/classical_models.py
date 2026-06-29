"""
Classical Machine Learning Baselines
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a classical ML model.
    """

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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

    return {
        "accuracy": accuracy,
        "matrix": matrix,
        "report": report
    }


def compare_classical_models(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Train multiple classical models.
    """

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

        "Support Vector Machine":
            SVC(),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )

    }

    results = {}

    print("\n" + "=" * 60)
    print("CLASSICAL MODEL COMPARISON")
    print("=" * 60)

    for name, model in models.items():

        print(f"\nTraining {name}...")

        result = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test
        )

        results[name] = result

        print(
            f"Accuracy : {result['accuracy']:.4f}"
        )

    return results