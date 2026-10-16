from __future__ import annotations

import numpy as np

from quantum_vision import HybridClassifier, QuantumEncoder, Quanvolution


def bars_and_stripes(n_samples: int, size: int = 4, noise: float = 0.15,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images, labels = [], []
    while len(images) < n_samples:
        pattern = rng.integers(0, 2, size=size)
        if pattern.min() == pattern.max():
            continue
        label = int(rng.integers(0, 2))
        image = np.tile(pattern, (size, 1)) if label == 0 else np.tile(pattern[:, None], (1, size))
        image = np.clip(image + rng.normal(0, noise, image.shape), 0, 1)
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)


def ascii_image(image: np.ndarray) -> list[str]:
    shades = " ░▒▓█"
    return ["".join(shades[int(v * (len(shades) - 1))] * 2 for v in row) for row in image]


def main() -> None:
    print("=" * 60)
    print("  Quantum Vision — Bars & Stripes")
    print("=" * 60)

    X, y = bars_and_stripes(200, seed=0)
    X_train, y_train, X_test, y_test = X[:160], y[:160], X[160:], y[160:]

    print("\n  Sample images (0 = vertical bars, 1 = horizontal stripes):")
    columns = [ascii_image(X[i]) for i in range(4)]
    for row in zip(*columns):
        print("    " + "    ".join(row))
    print("    " + "    ".join(f"label {y[i]} " for i in range(4)))

    info = QuantumEncoder.encoding_info(X[0])
    print("\n  Qubits needed to load one 4×4 image:")
    for name in ("amplitude_encoding", "FRQI", "NEQR", "angle_encoding"):
        print(f"    {name:20s} {info[name]['qubits']:3d} qubits")

    print("\n  1. Quanvolution — a 2×2 quantum filter slides over the image")
    quanv = Quanvolution(n_qubits=4, n_layers=1, patch_size=2, seed=1)
    filter_params = quanv.random_quantum_filter(seed=1)
    feature_map = quanv.quanvolve(X[0], filter_params, stride=2)
    print(f"    input {X[0].shape} → quantum feature map {feature_map.shape}")

    print("\n  2. Hybrid MLP → VQC → softmax, trained end-to-end with parameter-shift")
    model = HybridClassifier(input_dim=16, hidden_dim=8, n_qubits=4, n_layers=2,
                             n_classes=2, seed=0)
    model.train(X_train.reshape(len(X_train), -1), y_train, epochs=10, lr=0.5,
                batch_size=8, verbose=True)
    test_acc = model.accuracy(X_test.reshape(len(X_test), -1), y_test)
    print(f"\n    Test accuracy on {len(X_test)} unseen noisy images: {test_acc:.1%}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
