"""
Dataset Loading and Preprocessing
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.config import RANDOM_STATE, TEST_SIZE


def load_dataset():

    data = load_breast_cancer()

    X = data.data
    y = data.target

    return X, y


def preprocess_data(X, y, pca_components):

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    pca = PCA(
        n_components=pca_components
    )

    X = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )