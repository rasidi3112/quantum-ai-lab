"""
Quantum Image Encoding Schemes
================================

Implements several methods for encoding classical image data into quantum
states, enabling quantum processing of visual information.

Encoding methods:
    1. Amplitude Encoding: |ψ⟩ = Σ xᵢ|i⟩ / ||x||
       - Most compact: log₂(N) qubits for N pixels
       - Requires state preparation circuit

    2. Angle Encoding: ∏ᵢ RY(f(xᵢ))|0⟩
       - One qubit per feature
       - Simple circuit implementation

    3. Threshold (Binary) Encoding: |b₁b₂...bₙ⟩
       - Binary thresholding of pixel values
       - Simplest encoding

    4. FRQI (Flexible Representation of Quantum Images):
       - |I⟩ = (1/√N) Σᵢ (cos θᵢ|0⟩ + sin θᵢ|1⟩) ⊗ |i⟩
       - Encodes pixel intensities as rotation angles
       - log₂(N) + 1 qubits for N pixels

    5. NEQR (Novel Enhanced Quantum Representation):
       - Uses basis states to encode pixel gray levels
       - Better measurement retrieval than FRQI
       - log₂(N) + q qubits for N pixels with 2^q gray levels

References:
    - Le et al., "A Flexible Representation of Quantum Images" (2011)
    - Zhang et al., "NEQR: A Novel Enhanced Quantum Representation" (2013)
    - Schuld & Petruccione, "Supervised Learning with Quantum Computers" (2018)
"""

import numpy as np
from typing import Optional, Tuple


class QuantumEncoder:
    """
    Quantum image encoding with multiple strategies.

    Provides methods to encode classical image data into quantum states
    and decode quantum states back into classical images.

    Parameters
    ----------
    n_qubits : int, optional
        Default number of qubits for angle encoding.

    Examples
    --------
    >>> encoder = QuantumEncoder()
    >>> image = np.random.rand(4, 4)
    >>> state = encoder.amplitude_encoding(image)
    >>> recovered = encoder.decode_amplitude(state, (4, 4))
    """

    def __init__(self, n_qubits: Optional[int] = None):
        self.n_qubits = n_qubits

    # ------------------------------------------------------------------ #
    #  Amplitude Encoding                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def amplitude_encoding(image: np.ndarray) -> np.ndarray:
        """
        Encode a normalized, flattened image into state amplitudes.

        The image pixel values become the amplitudes of the quantum state:
            |ψ⟩ = Σᵢ xᵢ |i⟩ / ||x||

        The state vector is padded to the nearest power of 2.

        Parameters
        ----------
        image : np.ndarray
            Input image (any shape). Will be flattened.

        Returns
        -------
        np.ndarray
            Quantum state vector of length 2^⌈log₂(N)⌉ where
            N is the number of pixels.

        Notes
        -----
        - Requires ⌈log₂(N)⌉ qubits for N pixels.
        - The actual state preparation circuit is exponentially complex
          in the worst case, but this function represents the target state.
        """
        flat = image.flatten().astype(np.float64)
        n_pixels = len(flat)

        # Pad to nearest power of 2
        n_qubits = int(np.ceil(np.log2(max(n_pixels, 2))))
        dim = 2 ** n_qubits

        state = np.zeros(dim, dtype=np.complex128)
        state[:n_pixels] = flat

        # Normalize
        norm = np.linalg.norm(state)
        if norm > 1e-15:
            state /= norm

        return state

    # ------------------------------------------------------------------ #
    #  Angle Encoding                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def angle_encoding(image: np.ndarray, n_qubits: Optional[int] = None) -> np.ndarray:
        """
        Encode pixel values as rotation angles on individual qubits.

        Each qubit i is prepared as:
            RY(arctan(xᵢ)) |0⟩ = cos(arctan(xᵢ)/2)|0⟩ + sin(arctan(xᵢ)/2)|1⟩

        Parameters
        ----------
        image : np.ndarray
            Input image (any shape). Will be flattened.
        n_qubits : int, optional
            Number of qubits. If None, uses min(16, n_pixels).

        Returns
        -------
        np.ndarray
            Quantum state vector of length 2^n_qubits.

        Notes
        -----
        Angle encoding uses N qubits for N features, which is less compact
        than amplitude encoding but results in much simpler circuits.
        """
        flat = image.flatten().astype(np.float64)

        if n_qubits is None:
            n_qubits = min(16, len(flat))

        dim = 2 ** n_qubits

        # Start from |0...0⟩
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0

        # Apply RY(arctan(x_i)) to each qubit
        psi = state.reshape([2] * n_qubits)

        for i in range(min(len(flat), n_qubits)):
            angle = np.arctan(flat[i])
            c = np.cos(angle / 2)
            s = np.sin(angle / 2)
            ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
            psi = np.tensordot(ry, psi, axes=([1], [i]))
            psi = np.moveaxis(psi, 0, i)

        return psi.reshape(dim)

    # ------------------------------------------------------------------ #
    #  Threshold (Binary) Encoding                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def threshold_encoding(
        image: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """
        Binary encoding by thresholding pixel values.

        Pixels above the threshold are encoded as |1⟩, below as |0⟩.
        The resulting state is a computational basis state |b₁b₂...bₙ⟩.

        Parameters
        ----------
        image : np.ndarray
            Input image (any shape).
        threshold : float
            Threshold value (default 0.5).

        Returns
        -------
        np.ndarray
            Quantum state vector (a single basis state).
        """
        flat = image.flatten().astype(np.float64)
        n_pixels = len(flat)

        n_qubits = int(np.ceil(np.log2(max(n_pixels, 2))))
        dim = 2 ** n_qubits

        # Compute basis state index from binary encoding
        bits = (flat > threshold).astype(int)
        # Pad to n_qubits
        bits = np.pad(bits, (0, max(0, n_qubits - len(bits))))[:n_qubits]

        # Convert binary to index
        index = 0
        for b in bits:
            index = (index << 1) | b

        state = np.zeros(dim, dtype=np.complex128)
        state[index] = 1.0
        return state

    # ------------------------------------------------------------------ #
    #  FRQI — Flexible Representation of Quantum Images                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def FRQI(image: np.ndarray) -> np.ndarray:
        """
        Flexible Representation of Quantum Images (FRQI).

        Encodes an image as:
            |I⟩ = (1/√N) Σᵢ (cos θᵢ |0⟩ + sin θᵢ |1⟩) ⊗ |i⟩

        where θᵢ = (π/2) · xᵢ maps pixel intensity to angle, and |i⟩
        is the computational basis state encoding the pixel position.

        Parameters
        ----------
        image : np.ndarray
            Input image with values in [0, 1].

        Returns
        -------
        np.ndarray
            FRQI quantum state vector.

        Notes
        -----
        Uses ⌈log₂(N)⌉ + 1 qubits for N pixels.
        The first qubit encodes the color (intensity), and the remaining
        qubits encode the pixel position.
        """
        flat = image.flatten().astype(np.float64)
        # Normalize to [0, 1] if needed
        if flat.max() > 1.0:
            flat = flat / 255.0
        flat = np.clip(flat, 0, 1)

        n_pixels = len(flat)

        # Pad to power of 2
        n_pos_qubits = int(np.ceil(np.log2(max(n_pixels, 2))))
        N = 2 ** n_pos_qubits
        pixels = np.zeros(N)
        pixels[:n_pixels] = flat

        # Total qubits: 1 (color) + n_pos_qubits (position)
        n_total_qubits = 1 + n_pos_qubits
        dim = 2 ** n_total_qubits

        # Build FRQI state
        # |I⟩ = (1/√N) Σᵢ (cos θᵢ |0⟩ + sin θᵢ |1⟩) ⊗ |i⟩
        # In the full state vector, basis state index = color_bit * N + position
        state = np.zeros(dim, dtype=np.complex128)

        thetas = (np.pi / 2) * pixels

        for i in range(N):
            # |0⟩⊗|i⟩ → index = 0 * N + i = i
            state[i] += np.cos(thetas[i])
            # |1⟩⊗|i⟩ → index = 1 * N + i = N + i
            state[N + i] += np.sin(thetas[i])

        # Normalize
        state /= np.sqrt(N)

        return state

    # ------------------------------------------------------------------ #
    #  NEQR — Novel Enhanced Quantum Representation                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def NEQR(image: np.ndarray, n_gray_bits: int = 8) -> np.ndarray:
        """
        Novel Enhanced Quantum Representation (NEQR).

        Encodes each pixel's gray level in basis states:
            |I⟩ = (1/√N) Σᵢ |gᵢ⟩ ⊗ |i⟩

        where |gᵢ⟩ = |c₀c₁...c_{q-1}⟩ is the binary representation
        of the gray level.

        Parameters
        ----------
        image : np.ndarray
            Input image with values in [0, 1] (will be quantized to
            2^n_gray_bits levels).
        n_gray_bits : int
            Number of bits for gray level representation (default 8).

        Returns
        -------
        np.ndarray
            NEQR quantum state vector.

        Notes
        -----
        Uses ⌈log₂(N)⌉ + n_gray_bits qubits for N pixels with
        2^n_gray_bits gray levels. More qubits than FRQI but provides
        better information retrieval upon measurement.
        """
        flat = image.flatten().astype(np.float64)
        if flat.max() > 1.0:
            flat = flat / 255.0
        flat = np.clip(flat, 0, 1)

        n_pixels = len(flat)
        n_pos_qubits = int(np.ceil(np.log2(max(n_pixels, 2))))
        N = 2 ** n_pos_qubits

        # Quantize to integer gray levels
        max_gray = 2 ** n_gray_bits - 1
        gray_levels = np.zeros(N, dtype=int)
        gray_levels[:n_pixels] = np.round(flat * max_gray).astype(int)

        # Total qubits: n_gray_bits + n_pos_qubits
        n_total = n_gray_bits + n_pos_qubits
        dim = 2 ** n_total

        state = np.zeros(dim, dtype=np.complex128)

        # |I⟩ = (1/√N) Σᵢ |gᵢ⟩ ⊗ |i⟩
        # Index = gray_value * N + position
        G = 2 ** n_gray_bits
        for i in range(N):
            g = gray_levels[i]
            idx = g * N + i
            if idx < dim:
                state[idx] = 1.0

        # Normalize
        norm = np.linalg.norm(state)
        if norm > 1e-15:
            state /= norm

        return state

    # ------------------------------------------------------------------ #
    #  Decoding                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def decode_amplitude(
        state: np.ndarray, original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Reconstruct an image from an amplitude-encoded quantum state.

        Extracts the real part of the amplitudes and reshapes to the
        original image dimensions.

        Parameters
        ----------
        state : np.ndarray
            Quantum state vector.
        original_shape : tuple of int
            Shape of the original image.

        Returns
        -------
        np.ndarray
            Reconstructed image.

        Notes
        -----
        Due to normalization, the reconstructed image is proportional
        to the original but may differ in scale. The relative pixel
        intensities are preserved.
        """
        n_pixels = int(np.prod(original_shape))
        amplitudes = np.real(state[:n_pixels])

        # Un-normalize: find the scale factor
        norm = np.linalg.norm(amplitudes)
        if norm > 1e-15:
            amplitudes = amplitudes / norm

        return amplitudes.reshape(original_shape)

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compression_ratio(original_size: int, n_qubits: int) -> float:
        """
        Compute the quantum compression ratio.

        Ratio = original classical bits / quantum bits (qubits).

        Parameters
        ----------
        original_size : int
            Number of classical bits (e.g., 8 * n_pixels for 8-bit image).
        n_qubits : int
            Number of qubits used in the encoding.

        Returns
        -------
        float
            Compression ratio. Values > 1 indicate quantum advantage
            in terms of storage.

        Examples
        --------
        >>> # 64-pixel grayscale image (8 bits/pixel = 512 bits)
        >>> # Amplitude encoding: 6 qubits
        >>> QuantumEncoder.compression_ratio(512, 6)
        85.33...
        """
        return original_size / n_qubits

    @staticmethod
    def encoding_info(image: np.ndarray) -> dict:
        """
        Compute encoding statistics for a given image.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        dict
            Dictionary with encoding details for each method.
        """
        flat = image.flatten()
        n_pixels = len(flat)
        classical_bits = 8 * n_pixels  # assuming 8-bit pixels

        # Amplitude encoding
        amp_qubits = int(np.ceil(np.log2(max(n_pixels, 2))))

        # Angle encoding
        angle_qubits = n_pixels

        # FRQI
        frqi_qubits = int(np.ceil(np.log2(max(n_pixels, 2)))) + 1

        # NEQR (8-bit gray)
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
