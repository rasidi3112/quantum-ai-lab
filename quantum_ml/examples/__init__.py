"""
Iris Classification with Quantum ML
====================================

Demonstrates the VariationalClassifier on a subset of the Iris dataset
(binary classification: setosa vs. versicolor).

No external dataset dependencies — data is hardcoded from the original
Fisher Iris dataset.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from quantum_ml import VariationalClassifier, QuantumKernel


def load_iris_binary():
    """Load a subset of the Iris dataset for binary classification.

    Returns setosa (class 0) vs. versicolor (class 1) with 2 features
    (sepal length, sepal width) for efficiency.
    """
    # Iris setosa samples (sepal_length, sepal_width)
    setosa = np.array([
        [5.1, 3.5], [4.9, 3.0], [4.7, 3.2], [4.6, 3.1], [5.0, 3.6],
        [5.4, 3.9], [4.6, 3.4], [5.0, 3.4], [4.4, 2.9], [4.9, 3.1],
        [5.4, 3.7], [4.8, 3.4], [4.8, 3.0], [4.3, 3.0], [5.8, 4.0],
    ])

    # Iris versicolor samples (sepal_length, sepal_width)
    versicolor = np.array([
        [7.0, 3.2], [6.4, 3.2], [6.9, 3.1], [5.5, 2.3], [6.5, 2.8],
        [5.7, 2.8], [6.3, 3.3], [4.9, 2.4], [6.6, 2.9], [5.2, 2.7],
        [5.0, 2.0], [5.9, 3.0], [6.0, 2.2], [6.1, 2.9], [5.6, 2.9],
    ])

    X = np.vstack([setosa, versicolor])
    y = np.array([0] * len(setosa) + [1] * len(versicolor))

    # Normalize features to [0, π]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min) * np.pi

    return X_norm, y


def main():
    print("=" * 60)
    print("  Quantum ML — Iris Classification Demo")
    print("=" * 60)

    # Load data
    X, y = load_iris_binary()
    n_train = 20
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"\nDataset: {len(X)} samples (train: {n_train}, test: {len(X_test)})")
    print(f"Features: 2 (sepal length, sepal width)")
    print(f"Classes: setosa (0) vs versicolor (1)")

    # --- Quantum Kernel Classification ---
    print("\n" + "-" * 60)
    print("  1. Quantum Kernel Classification")
    print("-" * 60)

    qk = QuantumKernel(n_qubits=2, n_layers=1)
    print("\nComputing quantum kernel matrix...")
    K_train = qk.kernel_matrix(X_train)
    print(f"Kernel matrix shape: {K_train.shape}")
    print(f"Kernel matrix sample (top-left 4×4):")
    for row in K_train[:4, :4]:
        print("  " + "  ".join(f"{v:.3f}" for v in row))

    y_svm = np.where(y_train == 0, -1, 1)
    y_svm_test = np.where(y_test == 0, -1, 1)
    preds = qk.classify_svm(X_train, y_svm, X_test, alpha=0.1)
    acc = np.mean(np.sign(preds) == y_svm_test)
    print(f"\nQuantum Kernel SVM Accuracy: {acc:.1%}")

    # --- Variational Quantum Classifier ---
    print("\n" + "-" * 60)
    print("  2. Variational Quantum Classifier")
    print("-" * 60)

    clf = VariationalClassifier(n_qubits=2, n_layers=2, random_state=42)
    result = clf.train(X_train, y_train, epochs=50, lr=0.5, method="COBYLA")

    train_acc = clf.accuracy(X_train, y_train)
    test_acc = clf.accuracy(X_test, y_test)

    print(f"\n  Results:")
    print(f"    Train accuracy: {train_acc:.1%}")
    print(f"    Test accuracy:  {test_acc:.1%}")
    print(f"    Final loss:     {result['loss']:.6f}")
    print(f"    Iterations:     {result['n_iterations']}")

    # Print predictions
    print(f"\n  Test Predictions:")
    preds = clf.predict(X_test)
    for i, (pred, true) in enumerate(zip(preds, y_test)):
        status = "✓" if pred == true else "✗"
        label_pred = "setosa" if pred == 0 else "versicolor"
        label_true = "setosa" if true == 0 else "versicolor"
        print(f"    Sample {i+1}: predicted={label_pred:12s} true={label_true:12s} {status}")

    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
