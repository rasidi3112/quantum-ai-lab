"""
E91 (Ekert 1991) Entanglement-Based QKD Protocol
=================================================

The E91 protocol uses maximally entangled Bell pairs to distribute a shared
secret key.  Its security relies on Bell's theorem: any eavesdropper who
disturbs the entanglement will reduce the observed CHSH correlation below the
quantum-mechanical maximum of ``2√2 ≈ 2.828``, alerting Alice and Bob.

Bell state used
---------------
.. math::

    |\\Phi^+\\rangle = \\frac{1}{\\sqrt{2}} (|00\\rangle + |11\\rangle)

Measurement settings (Ekert's original choice)
----------------------------------------------
*  Bell state: ``|Ψ⁻⟩ = (|01⟩ − |10⟩) / √2``
*  Alice's angles:  ``{0, π/4, π/2}``       (labelled a₁, a₂, a₃)
*  Bob's angles:    ``{π/4, π/2, 3π/4}``    (labelled b₁, b₂, b₃)

Key extraction: pairs where Alice and Bob choose the *same* angle index
(a₁↔b₁ or a₃↔b₃) produce perfectly anti-correlated outcomes for |Ψ⁻⟩.
The remaining cross-pair combinations feed the CHSH test.

CHSH inequality
---------------
The CHSH-maximising combination for |Ψ⁻⟩:

.. math::

    S = E(a_2, b_1) - E(a_2, b_3) + E(a_3, b_1) + E(a_3, b_3)

With the angles above this gives |S| = 2√2 ≈ 2.828 for genuine entanglement.
Classical bound: |S| ≤ 2.  Quantum mechanics allows |S| ≤ 2√2.
A significant violation certifies genuine entanglement (no eavesdropper).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Basis states and Bell pair
# ---------------------------------------------------------------------------
KET_0 = np.array([1.0, 0.0], dtype=complex)
KET_1 = np.array([0.0, 1.0], dtype=complex)

# |Ψ⁻⟩ = (|01⟩ − |10⟩) / √2  — the singlet Bell state.
# With this state E(a,b) = -cos(a−b), which enables |S| = 2√2.
BELL_PSI_MINUS = (np.kron(KET_0, KET_1) - np.kron(KET_1, KET_0)) / np.sqrt(2)


# ---------------------------------------------------------------------------
# Helper: single-qubit rotation about Z-then-X (measurement in angle θ)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Statistics container
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# E91 Protocol
# ---------------------------------------------------------------------------

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

    # Ekert's measurement angles — chosen to maximise Bell violation with |Ψ⁻⟩
    ALICE_ANGLES = (0.0, np.pi / 4, np.pi / 2)            # a1, a2, a3
    BOB_ANGLES   = (np.pi / 4, np.pi / 2, 3 * np.pi / 4)  # b1, b2, b3

    # CHSH uses a2 and a3 vs b1 and b3:
    #   S = E(a2,b1) - E(a2,b3) + E(a3,b1) + E(a3,b3) ≈ −2√2  for |Ψ⁻⟩
    # (|S| > 2 certifies genuine quantum correlations)

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    # ----- entangled pair generation ----------------------------------------

    @staticmethod
    def generate_entangled_pairs(n: int) -> List[np.ndarray]:
        """Create *n* copies of the singlet Bell state |Ψ⁻⟩.

        Returns
        -------
        list of np.ndarray, each shape ``(4,)``
        """
        return [BELL_PSI_MINUS.copy() for _ in range(n)]

    # ----- measurement settings ---------------------------------------------

    @classmethod
    def choose_measurement_angles(cls) -> Tuple[Tuple[float, ...],
                                                  Tuple[float, ...]]:
        """Return the sets of measurement angles for Alice and Bob.

        Returns
        -------
        (alice_angles, bob_angles)
        """
        return cls.ALICE_ANGLES, cls.BOB_ANGLES

    # ----- single-pair measurement ------------------------------------------

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

        # Four joint projectors: |a±⟩⊗|b±⟩
        projectors = [
            np.kron(a_plus,  b_plus),   # (0, 0)
            np.kron(a_plus,  b_minus),  # (0, 1)
            np.kron(a_minus, b_plus),   # (1, 0)
            np.kron(a_minus, b_minus),  # (1, 1)
        ]

        probs = np.array([np.abs(np.vdot(p, pair)) ** 2 for p in projectors])
        probs /= probs.sum()  # normalise for numerical safety

        outcome_idx = self.rng.choice(4, p=probs)
        outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        return outcomes[outcome_idx]

    # ----- CHSH computation -------------------------------------------------

    @staticmethod
    def compute_chsh(correlations: Dict[Tuple[float, float], float]) -> float:
        """Compute the CHSH value *S* from pairwise correlations.

        Uses the standard CHSH-maximising angle combination for |Φ+⟩:

        .. math::

            S = E(a_2, b_1) - E(a_2, b_3) + E(a_3, b_1) + E(a_3, b_3)

        where ``a₂ = π/8``, ``a₃ = π/4``, ``b₁ = 0``, ``b₃ = −π/8``.
        For ideal entangled pairs this yields |S| = 2√2 ≈ 2.828,
        violating the classical bound of 2.

        Parameters
        ----------
        correlations : dict
            Mapping ``(angle_a, angle_b) → E``, where ``E`` is the
            expectation value of the product of ±1 outcomes.

        Returns
        -------
        float
            The CHSH parameter S.
        """
        a2, a3 = np.pi / 4, np.pi / 2
        b1, b3 = np.pi / 4, 3 * np.pi / 4

        def _E(a: float, b: float) -> float:
            key = (round(a, 10), round(b, 10))
            return correlations.get(key, 0.0)

        # For |Ψ⁻⟩: S ≈ -2√2 ≈ -2.828, so |S| > 2 signals true entanglement
        S = _E(a2, b1) - _E(a2, b3) + _E(a3, b1) + _E(a3, b3)
        return S

    @staticmethod
    def check_bell_violation(S: float) -> bool:
        """Return True if |S| > 2 (Bell inequality violated)."""
        return abs(S) > 2.0

    # ----- key extraction ---------------------------------------------------

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

    # ----- full protocol ----------------------------------------------------

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

        # Random angle choices for each pair
        a_choices = self.rng.integers(0, 3, size=n_pairs)
        b_choices = self.rng.integers(0, 3, size=n_pairs)

        # Containers
        key_results: List[Tuple[int, int]] = []
        corr_counts: Dict[Tuple[float, float], List[int]] = {}

        for i in range(n_pairs):
            aa = alice_angles[a_choices[i]]
            bb = bob_angles[b_choices[i]]
            oa, ob = self.measure_entangled(pairs[i], aa, bb)

            # Key-generation pairs: same angle index (a1↔b1 or a3↔b3)
            # These settings produce maximally-correlated outcomes for |Φ+⟩
            if a_choices[i] == b_choices[i] and a_choices[i] != 1:
                key_results.append((oa, ob))
            else:
                # Store correlation data for CHSH test
                key_pair = (round(aa, 10), round(bb, 10))
                if key_pair not in corr_counts:
                    corr_counts[key_pair] = []
                # Product of ±1 outcomes: map 0→+1, 1→−1
                product = (1 - 2 * oa) * (1 - 2 * ob)
                corr_counts[key_pair].append(product)

        # Compute correlations
        correlations_raw: Dict[Tuple[float, float], float] = {}
        for k, products in corr_counts.items():
            correlations_raw[k] = float(np.mean(products))

        # CHSH
        S = self.compute_chsh(correlations_raw)
        stats.chsh_value = S
        stats.bell_violated = self.check_bell_violation(S)
        stats.correlations = {str(k): v for k, v in correlations_raw.items()}

        # Key extraction
        alice_key, bob_key = self.extract_key(key_results)
        stats.alice_key = alice_key
        stats.bob_key = bob_key
        stats.n_key_bits = len(alice_key)

        return np.array(alice_key, dtype=int), stats
