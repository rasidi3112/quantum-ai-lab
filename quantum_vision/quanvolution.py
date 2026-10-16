"""Quanvolutional filters.

A small random or trainable circuit is slid over image patches like a
convolution kernel; each patch is angle-encoded and every qubit's measured
``P(0)`` becomes one output channel.

References
----------
Henderson, M. et al. "Quanvolutional neural networks: powering image
recognition with quantum circuits." Quantum Mach. Intell. 2, 2 (2020).
"""

from __future__ import annotations

import numpy as np


class Quanvolution:
    """Quantum convolution over square image patches.

    Parameters
    ----------
    n_qubits : int, default 4
        Qubits per filter; ``patch_size**2`` should not exceed it.
    n_layers : int, default 2
        Variational layers per filter.
    patch_size : int, default 2
        Side length of the square patch.
    seed : int, optional
        Seed for random filters.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        patch_size: int = 2,
        seed: int | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.dim = 2 ** n_qubits
        self.rng = np.random.default_rng(seed)

        self._I = np.eye(2, dtype=np.complex128)
        self._X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self._Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


    @staticmethod
    def _ry(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @staticmethod
    def _rz(phi: float) -> np.ndarray:
        return np.array([
            [np.exp(-1j * phi / 2), 0],
            [0, np.exp(1j * phi / 2)],
        ], dtype=np.complex128)

    def _apply_gate(self, state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
        psi = state.reshape([2] * self.n_qubits)
        psi = np.tensordot(gate, psi, axes=([1], [qubit]))
        psi = np.moveaxis(psi, 0, qubit)
        return psi.reshape(self.dim)

    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        n = self.n_qubits
        psi = state.reshape([2] * n)
        idx_1: list[slice | int] = [slice(None)] * n
        idx_1[control] = 1

        psi_1 = psi[tuple(idx_1)].copy()
        target_ax = target - (1 if target > control else 0)
        psi_1 = np.tensordot(self._X, psi_1, axes=([1], [target_ax]))
        psi_1 = np.moveaxis(psi_1, 0, target_ax)
        psi[tuple(idx_1)] = psi_1
        return psi.reshape(self.dim)

    def _z_expectation(self, state: np.ndarray, qubit: int) -> float:
        psi = state.reshape([2] * self.n_qubits)
        idx_0: list[slice | int] = [slice(None)] * self.n_qubits
        idx_0[qubit] = 0
        p0 = np.sum(np.abs(psi[tuple(idx_0)]) ** 2)
        z_exp = 2 * p0 - 1
        return float((z_exp + 1) / 2)


    def _run_circuit(self, pixel_values: np.ndarray, params: np.ndarray) -> np.ndarray:
        state = np.zeros(self.dim, dtype=np.complex128)
        state[0] = 1.0

        for i in range(min(len(pixel_values), self.n_qubits)):
            angle = np.pi * pixel_values[i]
            state = self._apply_gate(state, self._ry(angle), i)

        params_per_layer = 2 * self.n_qubits
        for layer in range(self.n_layers):
            start = layer * params_per_layer
            layer_params = params[start : start + params_per_layer]

            for i in range(self.n_qubits):
                state = self._apply_gate(state, self._ry(layer_params[i]), i)
                state = self._apply_gate(state, self._rz(layer_params[self.n_qubits + i]), i)

            if self.n_qubits > 1:
                for i in range(self.n_qubits):
                    state = self._apply_cnot(state, i, (i + 1) % self.n_qubits)

        features = np.array([self._z_expectation(state, i) for i in range(self.n_qubits)])
        return features


    def random_quantum_filter(
        self,
        n_qubits: int | None = None,
        n_layers: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Draw random filter angles uniformly from ``[0, 2 pi)``.

        Parameters
        ----------
        n_qubits, n_layers : int, optional
            Override the instance settings.
        seed : int, optional
            Seed for this filter only.

        Returns
        -------
        numpy.ndarray
        """
        nq = n_qubits or self.n_qubits
        nl = n_layers or self.n_layers
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        return rng.uniform(0, 2 * np.pi, size=nl * 2 * nq)

    def apply_filter(self, patch: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Run the filter circuit on one patch.

        Parameters
        ----------
        patch : numpy.ndarray
            Pixel values in ``[0, 1]``, encoded as ``RY(pi * p)``.
        params : numpy.ndarray
            Filter angles.

        Returns
        -------
        numpy.ndarray
            ``P(0)`` of each qubit, in ``[0, 1]``.
        """
        pixels = patch.flatten()[:self.n_qubits]
        if len(pixels) < self.n_qubits:
            pixels = np.pad(pixels, (0, self.n_qubits - len(pixels)))
        return self._run_circuit(pixels, params)

    def quanvolve(
        self,
        image: np.ndarray,
        filter_params: np.ndarray,
        stride: int = 2,
    ) -> np.ndarray:
        """Slide the filter over a 2-D image.

        Parameters
        ----------
        image : numpy.ndarray
            Grayscale image with values in ``[0, 1]``.
        filter_params : numpy.ndarray
            Filter angles.
        stride : int, default 2
            Step between patches.

        Returns
        -------
        numpy.ndarray
            Feature map of shape ``(H_out, W_out, n_qubits)``.
        """
        H, W = image.shape
        ps = self.patch_size

        H_out = (H - ps) // stride + 1
        W_out = (W - ps) // stride + 1

        output = np.zeros((H_out, W_out, self.n_qubits))

        for i in range(H_out):
            for j in range(W_out):
                r = i * stride
                c = j * stride
                patch = image[r : r + ps, c : c + ps]
                output[i, j, :] = self.apply_filter(patch, filter_params)

        return output

    def trainable_filter(
        self,
        n_qubits: int | None = None,
        n_layers: int | None = None,
    ) -> np.ndarray:
        """Draw small initial angles ``N(0, 0.1^2)`` for a trainable filter."""
        nq = n_qubits or self.n_qubits
        nl = n_layers or self.n_layers
        return self.rng.normal(0, 0.1, size=nl * 2 * nq)

    def multi_filter(
        self,
        image: np.ndarray,
        n_filters: int,
        params_list: list[np.ndarray],
        stride: int = 2,
    ) -> np.ndarray:
        """Apply several filters and stack their feature maps.

        Parameters
        ----------
        image : numpy.ndarray
            Grayscale image.
        n_filters : int
            Number of filters to apply.
        params_list : list of numpy.ndarray
            Angles of each filter.
        stride : int, default 2
            Step between patches.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(H_out, W_out, n_filters * n_qubits)``.

        Raises
        ------
        ValueError
            If fewer than ``n_filters`` parameter sets are given.
        """
        if len(params_list) < n_filters:
            raise ValueError(
                f"Expected {n_filters} parameter sets, got {len(params_list)}"
            )

        feature_maps = []
        for k in range(n_filters):
            fm = self.quanvolve(image, params_list[k], stride=stride)
            feature_maps.append(fm)

        return np.concatenate(feature_maps, axis=-1)
