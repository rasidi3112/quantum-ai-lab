import numpy as np
from typing import Optional, Tuple


class QuantumEncoder:
    def __init__(self, n_qubits: Optional[int] = None):
        self.n_qubits = n_qubits


    @staticmethod
    def amplitude_encoding(image: np.ndarray) -> np.ndarray:
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
    def angle_encoding(image: np.ndarray, n_qubits: Optional[int] = None) -> np.ndarray:
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
        flat = image.flatten().astype(np.float64)
        if flat.max() > 1.0:
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
        flat = image.flatten().astype(np.float64)
        if flat.max() > 1.0:
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

        G = 2 ** n_gray_bits
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
        state: np.ndarray, original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        n_pixels = int(np.prod(original_shape))
        amplitudes = np.real(state[:n_pixels])

        norm = np.linalg.norm(amplitudes)
        if norm > 1e-15:
            amplitudes = amplitudes / norm

        return amplitudes.reshape(original_shape)


    @staticmethod
    def compression_ratio(original_size: int, n_qubits: int) -> float:
        return original_size / n_qubits

    @staticmethod
    def encoding_info(image: np.ndarray) -> dict:
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
