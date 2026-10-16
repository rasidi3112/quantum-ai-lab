"""Image encodings for quantum circuits.

Amplitude encoding needs only ``log2(N)`` qubits for ``N`` pixels, FRQI adds
one color qubit, NEQR stores gray levels in a binary register, and angle
encoding uses one qubit per pixel.

References
----------
Le, P. Q., Dong, F. & Hirota, K. "A flexible representation of quantum images
for polynomial preparation, image compression, and processing operations."
Quantum Inf. Process. 10, 63 (2011).
Zhang, Y. et al. "NEQR: a novel enhanced quantum representation of digital
images." Quantum Inf. Process. 12, 2833 (2013).
"""

from __future__ import annotations

import numpy as np


class QuantumEncoder:
    """Collection of image-to-statevector encodings.

    Parameters
    ----------
    n_qubits : int, optional
        Stored for reference; the encoders size their registers automatically.
    """

    def __init__(self, n_qubits: int | None = None) -> None:
        self.n_qubits = n_qubits


    @staticmethod
    def amplitude_encoding(image: np.ndarray) -> np.ndarray:
        """Store normalized pixel values in the amplitudes.

        Parameters
        ----------
        image : numpy.ndarray
            Image of any shape.

        Returns
        -------
        numpy.ndarray
            State on ``ceil(log2(n_pixels))`` qubits.
        """
        flat = image.flatten().astype(np.float64)
        n_pixels = len(flat)

        n_qubits = int(np.ceil(np.log2(max(n_pixels, 2))))
        dim = 2 ** n_qubits

        state = np.zeros(dim, dtype=np.complex128)
        state[:n_pixels] = flat

        norm = np.linalg.norm(state)
        if norm > 1e-15:
            state /= norm

        return state


    @staticmethod
    def angle_encoding(image: np.ndarray, n_qubits: int | None = None) -> np.ndarray:
        """Encode pixels as ``RY(arctan(p))`` rotations, one qubit per pixel.

        Parameters
        ----------
        image : numpy.ndarray
            Image of any shape.
        n_qubits : int, optional
            Number of qubits; defaults to ``min(16, n_pixels)``.

        Returns
        -------
        numpy.ndarray
        """
        flat = image.flatten().astype(np.float64)

        if n_qubits is None:
            n_qubits = min(16, len(flat))

        dim = 2 ** n_qubits

        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0

        psi = state.reshape([2] * n_qubits)

        for i in range(min(len(flat), n_qubits)):
            angle = np.arctan(flat[i])
            c = np.cos(angle / 2)
            s = np.sin(angle / 2)
            ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
            psi = np.tensordot(ry, psi, axes=([1], [i]))
            psi = np.moveaxis(psi, 0, i)

        return psi.reshape(dim)


    @staticmethod
    def threshold_encoding(
        image: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """Basis state built from thresholded pixels.

        Parameters
        ----------
        image : numpy.ndarray
            Image of any shape.
        threshold : float, default 0.5
            Pixels above it map to 1.

        Returns
        -------
        numpy.ndarray
            A computational basis state on ``ceil(log2(n_pixels))`` qubits whose
            bits are the first thresholded pixels.
        """
        flat = image.flatten().astype(np.float64)
        n_pixels = len(flat)

        n_qubits = int(np.ceil(np.log2(max(n_pixels, 2))))
        dim = 2 ** n_qubits

        bits = (flat > threshold).astype(int)
        bits = np.pad(bits, (0, max(0, n_qubits - len(bits))))[:n_qubits]

        index = 0
        for b in bits:
            index = (index << 1) | b

        state = np.zeros(dim, dtype=np.complex128)
        state[index] = 1.0
        return state


    @staticmethod
    def FRQI(image: np.ndarray) -> np.ndarray:
        """Flexible representation of quantum images.

        ``|I> = 2^(-n/2) sum_i (cos(theta_i)|0> + sin(theta_i)|1>) |i>`` with
        ``theta_i = (pi/2) * pixel_i``.

        Parameters
        ----------
        image : numpy.ndarray
            Pixel values in ``[0, 1]`` (values above 1 are treated as 8-bit).

        Returns
        -------
        numpy.ndarray
            State on ``1 + ceil(log2(n_pixels))`` qubits.
        """
        flat = image.flatten().astype(np.float64)
        if np.max(flat) > 1.0:
            flat = flat / 255.0
        flat = np.clip(flat, 0, 1)

        n_pixels = len(flat)

        n_pos_qubits = int(np.ceil(np.log2(max(n_pixels, 2))))
        N = 2 ** n_pos_qubits
        pixels = np.zeros(N)
        pixels[:n_pixels] = flat

        n_total_qubits = 1 + n_pos_qubits
        dim = 2 ** n_total_qubits

        state = np.zeros(dim, dtype=np.complex128)

        thetas = (np.pi / 2) * pixels

        for i in range(N):
            state[i] += np.cos(thetas[i])
            state[N + i] += np.sin(thetas[i])

        state /= np.sqrt(N)

        return state


    @staticmethod
    def NEQR(image: np.ndarray, n_gray_bits: int = 8) -> np.ndarray:
        """Novel enhanced quantum representation with a binary gray-level register.

        Parameters
        ----------
        image : numpy.ndarray
            Pixel values in ``[0, 1]`` (values above 1 are treated as 8-bit).
        n_gray_bits : int, default 8
            Bits of gray-level resolution.

        Returns
        -------
        numpy.ndarray
            State on ``n_gray_bits + ceil(log2(n_pixels))`` qubits.
        """
        flat = image.flatten().astype(np.float64)
        if np.max(flat) > 1.0:
            flat = flat / 255.0
        flat = np.clip(flat, 0, 1)

        n_pixels = len(flat)
        n_pos_qubits = int(np.ceil(np.log2(max(n_pixels, 2))))
        N = 2 ** n_pos_qubits

        max_gray = 2 ** n_gray_bits - 1
        gray_levels = np.zeros(N, dtype=int)
        gray_levels[:n_pixels] = np.round(flat * max_gray).astype(int)

        n_total = n_gray_bits + n_pos_qubits
        dim = 2 ** n_total

        state = np.zeros(dim, dtype=np.complex128)

        for i in range(N):
            g = gray_levels[i]
            idx = g * N + i
            if idx < dim:
                state[idx] = 1.0

        norm = np.linalg.norm(state)
        if norm > 1e-15:
            state /= norm

        return state


    @staticmethod
    def decode_amplitude(
        state: np.ndarray, original_shape: tuple[int, ...]
    ) -> np.ndarray:
        """Recover a normalized image from an amplitude-encoded state.

        Parameters
        ----------
        state : numpy.ndarray
            Amplitude-encoded state.
        original_shape : tuple of int
            Image shape.

        Returns
        -------
        numpy.ndarray
            The image divided by its Euclidean norm.
        """
        n_pixels = int(np.prod(original_shape))
        amplitudes = np.real(state[:n_pixels])

        norm = np.linalg.norm(amplitudes)
        if norm > 1e-15:
            amplitudes = amplitudes / norm

        return amplitudes.reshape(original_shape)


    @staticmethod
    def compression_ratio(original_size: int, n_qubits: int) -> float:
        """Pixels per qubit, ``original_size / n_qubits``."""
        return original_size / n_qubits

    @staticmethod
    def encoding_info(image: np.ndarray) -> dict:
        """Qubit counts of every encoding for an image.

        Parameters
        ----------
        image : numpy.ndarray
            Image of any shape.

        Returns
        -------
        dict
            Pixel count, classical bit count (8 bits per pixel), and the qubit count
            and compression ratio of each encoding.
        """
        flat = image.flatten()
        n_pixels = len(flat)
        classical_bits = 8 * n_pixels

        amp_qubits = int(np.ceil(np.log2(max(n_pixels, 2))))

        angle_qubits = n_pixels

        frqi_qubits = int(np.ceil(np.log2(max(n_pixels, 2)))) + 1

        neqr_qubits = int(np.ceil(np.log2(max(n_pixels, 2)))) + 8

        return {
            "n_pixels": n_pixels,
            "classical_bits": classical_bits,
            "amplitude_encoding": {
                "qubits": amp_qubits,
                "compression_ratio": classical_bits / amp_qubits,
            },
            "angle_encoding": {
                "qubits": angle_qubits,
                "compression_ratio": classical_bits / angle_qubits,
            },
            "FRQI": {
                "qubits": frqi_qubits,
                "compression_ratio": classical_bits / frqi_qubits,
            },
            "NEQR": {
                "qubits": neqr_qubits,
                "compression_ratio": classical_bits / neqr_qubits,
            },
        }
