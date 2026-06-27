"""
Quanvolutional Neural Network Layer
=====================================

Implements quanvolutional filters — quantum circuits that replace classical
convolutional kernels for image feature extraction. A small image patch
(e.g. 2×2 pixels) is encoded into qubit rotations, processed through a
parameterized quantum circuit, and measured to produce output features.

Architecture for a 2×2 patch (4 pixels → 4 qubits):

    |0⟩ ─ RY(π·p₀) ─ [Variational Layer(s)] ─ Measure → f₀
    |0⟩ ─ RY(π·p₁) ─ [Variational Layer(s)] ─ Measure → f₁
    |0⟩ ─ RY(π·p₂) ─ [Variational Layer(s)] ─ Measure → f₂
    |0⟩ ─ RY(π·p₃) ─ [Variational Layer(s)] ─ Measure → f₃

Where pᵢ are normalized pixel values in [0, 1] and fᵢ is the expectation
value of the Z operator on qubit i (mapped to [0, 1]).

References:
    - Henderson et al., "Quanvolutional Neural Networks:
      Powering Image Recognition with Quantum Circuits" (2020)
"""

import numpy as np
from typing import Optional, List, Tuple


class Quanvolution:
    """
    Quanvolutional filter using parameterized quantum circuits.

    Slides a quantum circuit across an image, encoding patches into
    quantum states and extracting features from measurements.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must match patch_size² for square patches).
    n_layers : int
        Number of variational layers in the circuit.
    patch_size : int
        Side length of the square patch (default 2, giving 2×2=4 pixels).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> qconv = Quanvolution(n_qubits=4, n_layers=2, seed=42)
    >>> image = np.random.rand(8, 8)
    >>> params = qconv.random_quantum_filter()
    >>> features = qconv.quanvolve(image, params, stride=2)
    >>> features.shape  # (4, 4, 4) for 8x8 image, stride=2, 4 qubits
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        patch_size: int = 2,
        seed: Optional[int] = None,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.dim = 2 ** n_qubits
        self._rng = np.random.default_rng(seed)

        # Pre-build gate matrices
        self._I = np.eye(2, dtype=np.complex128)
        self._X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self._Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # ------------------------------------------------------------------ #
    #  Gate primitives                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ry(theta: float) -> np.ndarray:
        """RY(θ) rotation gate."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @staticmethod
    def _rz(phi: float) -> np.ndarray:
        """RZ(φ) rotation gate."""
        return np.array([
            [np.exp(-1j * phi / 2), 0],
            [0, np.exp(1j * phi / 2)],
        ], dtype=np.complex128)

    def _apply_gate(self, state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
        """Apply single-qubit gate using tensor reshaping."""
        psi = state.reshape([2] * self.n_qubits)
        psi = np.tensordot(gate, psi, axes=([1], [qubit]))
        psi = np.moveaxis(psi, 0, qubit)
        return psi.reshape(self.dim)

    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT gate between control and target qubits."""
        n = self.n_qubits
        psi = state.reshape([2] * n)
        idx_1 = [slice(None)] * n
        idx_1[control] = 1

        psi_1 = psi[tuple(idx_1)].copy()
        target_ax = target - (1 if target > control else 0)
        psi_1 = np.tensordot(self._X, psi_1, axes=([1], [target_ax]))
        psi_1 = np.moveaxis(psi_1, 0, target_ax)
        psi[tuple(idx_1)] = psi_1
        return psi.reshape(self.dim)

    def _z_expectation(self, state: np.ndarray, qubit: int) -> float:
        """
        Compute ⟨ψ|Z_i|ψ⟩ for qubit i.

        Maps result from [-1, 1] to [0, 1] for use as a feature.
        """
        psi = state.reshape([2] * self.n_qubits)
        # Prob of qubit being |0⟩
        idx_0 = [slice(None)] * self.n_qubits
        idx_0[qubit] = 0
        p0 = np.sum(np.abs(psi[tuple(idx_0)]) ** 2)
        # ⟨Z⟩ = p(0) - p(1) = 2*p(0) - 1
        z_exp = 2 * p0 - 1
        # Map to [0, 1]
        return float((z_exp + 1) / 2)

    # ------------------------------------------------------------------ #
    #  Circuit execution                                                  #
    # ------------------------------------------------------------------ #

    def _run_circuit(self, pixel_values: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Run the quanvolutional circuit on a set of pixel values.

        Parameters
        ----------
        pixel_values : np.ndarray
            Normalized pixel values (0 to 1) of shape (n_qubits,).
        params : np.ndarray
            Circuit parameters of shape (n_layers * 2 * n_qubits,).

        Returns
        -------
        np.ndarray
            Feature vector of shape (n_qubits,), values in [0, 1].
        """
        # Initialize |0...0⟩
        state = np.zeros(self.dim, dtype=np.complex128)
        state[0] = 1.0

        # Encode pixel values as RY rotations
        for i in range(min(len(pixel_values), self.n_qubits)):
            angle = np.pi * pixel_values[i]
            state = self._apply_gate(state, self._ry(angle), i)

        # Apply variational layers
        params_per_layer = 2 * self.n_qubits
        for layer in range(self.n_layers):
            start = layer * params_per_layer
            layer_params = params[start : start + params_per_layer]

            # RY-RZ rotations
            for i in range(self.n_qubits):
                state = self._apply_gate(state, self._ry(layer_params[i]), i)
                state = self._apply_gate(state, self._rz(layer_params[self.n_qubits + i]), i)

            # CNOT ring
            if self.n_qubits > 1:
                for i in range(self.n_qubits):
                    state = self._apply_cnot(state, i, (i + 1) % self.n_qubits)

        # Measure: expectation of Z on each qubit → feature
        features = np.array([self._z_expectation(state, i) for i in range(self.n_qubits)])
        return features

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def random_quantum_filter(
        self,
        n_qubits: Optional[int] = None,
        n_layers: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate random circuit parameters for a quanvolutional filter.

        Parameters
        ----------
        n_qubits : int, optional
            Number of qubits (default: self.n_qubits).
        n_layers : int, optional
            Number of layers (default: self.n_layers).
        seed : int, optional
            Random seed for this specific filter.

        Returns
        -------
        np.ndarray
            Random parameters of shape (n_layers * 2 * n_qubits,).
        """
        nq = n_qubits or self.n_qubits
        nl = n_layers or self.n_layers
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        return rng.uniform(0, 2 * np.pi, size=nl * 2 * nq)

    def apply_filter(self, patch: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Apply a quanvolutional filter to a single image patch.

        Encodes the patch pixels into quantum rotations, runs the
        variational circuit, and measures expectation values.

        Parameters
        ----------
        patch : np.ndarray
            Image patch of shape (patch_size, patch_size) with values
            in [0, 1].
        params : np.ndarray
            Filter parameters.

        Returns
        -------
        np.ndarray
            Feature vector of shape (n_qubits,).
        """
        # Flatten patch to pixel vector
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
        """
        Apply quanvolutional filter across an entire image.

        Slides a (patch_size × patch_size) window across the image with
        the given stride, applying the quantum filter to each patch.

        Parameters
        ----------
        image : np.ndarray
            2D grayscale image of shape (H, W) with values in [0, 1].
        filter_params : np.ndarray
            Quantum filter parameters.
        stride : int
            Stride for the sliding window.

        Returns
        -------
        np.ndarray
            Feature map of shape (H_out, W_out, n_qubits) where
            H_out = (H - patch_size) // stride + 1, similarly for W_out.
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
        n_qubits: Optional[int] = None,
        n_layers: Optional[int] = None,
    ) -> np.ndarray:
        """
        Create trainable filter parameters initialized near zero.

        Parameters
        ----------
        n_qubits : int, optional
            Number of qubits (default: self.n_qubits).
        n_layers : int, optional
            Number of layers (default: self.n_layers).

        Returns
        -------
        np.ndarray
            Small-valued initial parameters.
        """
        nq = n_qubits or self.n_qubits
        nl = n_layers or self.n_layers
        return self._rng.normal(0, 0.1, size=nl * 2 * nq)

    def multi_filter(
        self,
        image: np.ndarray,
        n_filters: int,
        params_list: List[np.ndarray],
        stride: int = 2,
    ) -> np.ndarray:
        """
        Apply multiple quanvolutional filters to an image.

        Concatenates the feature maps from each filter along the channel
        dimension.

        Parameters
        ----------
        image : np.ndarray
            2D grayscale image of shape (H, W).
        n_filters : int
            Number of filters to apply.
        params_list : list of np.ndarray
            List of parameter arrays, one per filter.
        stride : int
            Stride for the sliding window.

        Returns
        -------
        np.ndarray
            Combined feature map of shape
            (H_out, W_out, n_filters * n_qubits).
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
