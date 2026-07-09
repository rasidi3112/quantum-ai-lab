"""
E91 (Ekert 1991) Entanglement-Based QKD Protocol
=================================================

The E91 protocol uses maximally entangled Bell pairs to distribute a shared
secret key.  Its security relies on Bell's theorem: any eavesdropper who
disturbs the entanglement will reduce the observed CHSH correlation below the
quantum-mechanical maximum of ``2√2 ≈ 2.828``, alerting Alice and Bob.

Bell state
----------
.. math::

    |\\Psi^-\\rangle = \\frac{1}{\\sqrt{2}} (|01\\rangle - |10\\rangle)

Measurement settings — 5-angle protocol
---------------------------------------
*  Bell state:     ``|Ψ⁻⟩ = (|01⟩ − |10⟩) / √2``  (singlet)
*  Alice's angles: ``{π/2, 0, 3π/8}``   (a₁, a₂, a_k)
*  Bob's angles:   ``{π/4, −π/4, 3π/8}``  (b₁, b₂, b_k)

The angle sets are chosen so that key-generation pairs and CHSH pairs
are **completely disjoint**:

* **Key pair** (index 2 for both): Alice=3π/8, Bob=3π/8.
  For |Ψ⁻⟩, ``E(θ, θ) = −1`` (perfect anti-correlation).
  Bob flips his bit → 100 % key agreement.
* **CHSH pairs** (Alice index 0 or 1, Bob index 0 or 1):
  The four combinations ``{π/2, 0} × {π/4, −π/4}`` feed the CHSH test.
* **Wasted pairs**: the remaining four cross-index combinations
  are discarded (~44 % of all pairs).

CHSH inequality
---------------
.. math::

    S = E(a_1, b_1) - E(a_1, b_2) + E(a_2, b_1) + E(a_2, b_2)

where ``a₁=π/2``, ``a₂=0``, ``b₁=π/4``, ``b₂=−π/4``.
For |Ψ⁻⟩ this evaluates to ``S = −2√2 ≈ −2.828``
(the quantum-mechanical maximum), certifying genuine entanglement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

KET_0 = np.array([1.0, 0.0], dtype=complex)
KET_1 = np.array([0.0, 1.0], dtype=complex)

BELL_PSI_MINUS = (np.kron(KET_0, KET_1) - np.kron(KET_1, KET_0)) / np.sqrt(2)


def _rotation_basis_vectors(theta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return the two eigenstates of the observable σ_θ = cos(θ)σ_z + sin(θ)σ_x.

    These correspond to measuring a qubit along direction θ in the XZ-plane
    of the Bloch sphere.

    Parameters
    ----------
    theta : float
        Measurement angle (radians).

    Returns
    -------
    (|+θ⟩, |−θ⟩) : tuple of 2-component complex arrays
        ``|+θ⟩`` has eigenvalue +1, ``|−θ⟩`` has eigenvalue −1.
    """
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)
    plus_theta  = np.array([ cos, sin], dtype=complex)
    minus_theta = np.array([-sin, cos], dtype=complex)
    return plus_theta, minus_theta


@dataclass
class E91Statistics:
    """Diagnostics for an E91 protocol run."""
    n_pairs: int = 0
    n_key_bits: int = 0
    chsh_value: float = 0.0
    bell_violated: bool = False
    correlations: Dict[str, float] = field(default_factory=dict)
    alice_key: List[int] = field(default_factory=list)
    bob_key: List[int] = field(default_factory=list)


class E91Protocol:
    """Simulation of the E91 entanglement-based QKD protocol.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.

    Examples
    --------
    >>> proto = E91Protocol(seed=42)
    >>> key, stats = proto.run_protocol(1000)
    >>> print(f"CHSH S = {stats.chsh_value:.4f}, Bell violated = {stats.bell_violated}")
    """

    ALICE_ANGLES = (np.pi / 2, 0.0,           3 * np.pi / 8)
    BOB_ANGLES   = (np.pi / 4, -np.pi / 4,    3 * np.pi / 8)

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)


    @staticmethod
    def generate_entangled_pairs(n: int) -> List[np.ndarray]:
        """Create *n* copies of the singlet Bell state |Ψ⁻⟩.

        Returns
        -------
        list of np.ndarray, each shape ``(4,)``
        """
        return [BELL_PSI_MINUS.copy() for _ in range(n)]


    @classmethod
    def choose_measurement_angles(cls) -> Tuple[Tuple[float, ...],
                                                  Tuple[float, ...]]:
        """Return the sets of measurement angles for Alice and Bob.

        Returns
        -------
        (alice_angles, bob_angles)
        """
        return cls.ALICE_ANGLES, cls.BOB_ANGLES


    def measure_entangled(self, pair: np.ndarray,
                          angle_a: float,
                          angle_b: float) -> Tuple[int, int]:
        """Measure both halves of a Bell pair in chosen angles.

        Implements the Born rule on the 4-dimensional statevector.

        Parameters
        ----------
        pair : np.ndarray, shape ``(4,)``
            Two-qubit Bell state.
        angle_a, angle_b : float
            Measurement angles (radians) for Alice and Bob respectively.

        Returns
        -------
        (outcome_a, outcome_b) : tuple of int
            Each is 0 (+1 eigenvalue) or 1 (−1 eigenvalue).
        """
        a_plus, a_minus = _rotation_basis_vectors(angle_a)
        b_plus, b_minus = _rotation_basis_vectors(angle_b)

        projectors = [
            np.kron(a_plus,  b_plus),
            np.kron(a_plus,  b_minus),
            np.kron(a_minus, b_plus),
            np.kron(a_minus, b_minus),
        ]

        probs = np.array([np.abs(np.vdot(p, pair)) ** 2 for p in projectors])
        probs /= probs.sum()

        outcome_idx = self.rng.choice(4, p=probs)
        outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        return outcomes[outcome_idx]


    @staticmethod
    def compute_chsh(correlations: Dict[Tuple[float, float], float]) -> float:
        """Compute the CHSH parameter *S* using the optimal 4-angle configuration.

        For the 5-angle protocol with |Ψ⁻⟩ and
        ``Alice={π/2, 0, 3π/8}``, ``Bob={π/4, −π/4, 3π/8}``:

        .. math::

            S = E(a_1, b_1) - E(a_1, b_2) + E(a_2, b_1) + E(a_2, b_2)

        where ``a₁=π/2``, ``a₂=0``, ``b₁=π/4``, ``b₂=−π/4``.

        Each correlation ``E(a,b) = −cos(a−b)`` for |Ψ⁻⟩, giving:

        * ``E(π/2, π/4)  = −cos(π/4) = −1/√2``
        * ``E(π/2, −π/4) = −cos(3π/4) = +1/√2``
        * ``E(0,   π/4)  = −cos(−π/4) = −1/√2``
        * ``E(0,   −π/4) = −cos(π/4)  = −1/√2``

        ``S = (−1/√2) − (+1/√2) + (−1/√2) + (−1/√2) = −2√2 ≈ −2.828``

        Parameters
        ----------
        correlations : dict
            Mapping ``(angle_a, angle_b) → E``, where ``E`` is the
            expectation value of the product of ±1 outcomes.

        Returns
        -------
        float
            The CHSH parameter S.  Ideal value: ``−2√2 ≈ −2.828``.
        """
        a1, a2 = np.pi / 2, 0.0
        b1, b2 = np.pi / 4, -np.pi / 4

        def _E(a: float, b: float) -> float:
            k = (round(a, 10), round(b, 10))
            return correlations.get(k, 0.0)

        S = _E(a1, b1) - _E(a1, b2) + _E(a2, b1) + _E(a2, b2)
        return S

    @staticmethod
    def check_bell_violation(S: float) -> bool:
        """Return True if |S| > 2 (Bell inequality violated)."""
        return abs(S) > 2.0


    @staticmethod
    def extract_key(results: List[Tuple[int, int]]) -> Tuple[List[int],
                                                               List[int]]:
        """Extract correlated key bits from matching-basis measurements.

        Parameters
        ----------
        results : list of (outcome_a, outcome_b)

        Returns
        -------
        (alice_key, bob_key) : tuple of list[int]
        """
        alice_key = [a for a, _ in results]
        bob_key   = [b for _, b in results]
        return alice_key, bob_key


    def run_protocol(self, n_pairs: int = 1000
                     ) -> Tuple[np.ndarray, E91Statistics]:
        """Run the full E91 QKD protocol simulation.

        Parameters
        ----------
        n_pairs : int
            Number of entangled pairs to distribute.

        Returns
        -------
        (key, stats) : tuple
        """
        stats = E91Statistics(n_pairs=n_pairs)
        pairs = self.generate_entangled_pairs(n_pairs)
        alice_angles, bob_angles = self.choose_measurement_angles()

        a_choices = self.rng.integers(0, 3, size=n_pairs)
        b_choices = self.rng.integers(0, 3, size=n_pairs)

        key_results: List[Tuple[int, int]] = []
        corr_counts: Dict[Tuple[float, float], List[int]] = {}

        for i in range(n_pairs):
            aa = alice_angles[a_choices[i]]
            bb = bob_angles[b_choices[i]]
            oa, ob = self.measure_entangled(pairs[i], aa, bb)

            ai, bi = int(a_choices[i]), int(b_choices[i])
            if ai == 2 and bi == 2:
                key_results.append((oa, 1 - ob))
            elif ai in (0, 1) and bi in (0, 1):
                k = (round(aa, 10), round(bb, 10))
                if k not in corr_counts:
                    corr_counts[k] = []
                corr_counts[k].append((1 - 2 * oa) * (1 - 2 * ob))

        correlations_raw: Dict[Tuple[float, float], float] = {}
        for k, products in corr_counts.items():
            correlations_raw[k] = float(np.mean(products))

        S = self.compute_chsh(correlations_raw)
        stats.chsh_value = S
        stats.bell_violated = self.check_bell_violation(S)
        stats.correlations = {str(k): v for k, v in correlations_raw.items()}

        alice_key, bob_key = self.extract_key(key_results)
        stats.alice_key = alice_key
        stats.bob_key = bob_key
        stats.n_key_bits = len(alice_key)

        return np.array(alice_key, dtype=int), stats
