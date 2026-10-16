from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from quantum_chaos import KickedTop, LevelSpacing, LyapunovEstimator


def test_kicked_top_floquet_is_unitary():
    top = KickedTop(j=10, k=3.0)
    U = top.floquet_operator
    assert np.allclose(U @ U.conj().T, np.eye(top.dim))
    assert np.allclose(np.abs(top.floquet_eigenvalues), 1.0)


def test_angular_momentum_algebra():
    top = KickedTop(j=3.5)
    Jx, Jy, Jz = top.Jx, top.Jy, top.Jz
    assert np.allclose(Jx @ Jy - Jy @ Jx, 1j * Jz)
    casimir = Jx @ Jx + Jy @ Jy + Jz @ Jz
    assert np.allclose(casimir, 3.5 * 4.5 * np.eye(top.dim))


def test_coherent_state_points_along_bloch_vector():
    top = KickedTop(j=20)
    theta, phi = 1.1, 0.4
    jx, jy, jz = top.expectation_values(top.coherent_state(theta, phi))
    assert np.linalg.norm(top.coherent_state(theta, phi)) == pytest.approx(1.0)
    assert jz == pytest.approx(np.cos(theta), abs=1e-10)
    assert np.hypot(jx, jy) == pytest.approx(np.sin(theta), abs=1e-10)


def test_kicked_top_rejects_invalid_spin():
    with pytest.raises(ValueError):
        KickedTop(j=1.3)


def test_ratio_statistic_distinguishes_goe_from_poisson():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(600, 600))
    goe = LevelSpacing.ratio_statistic(np.linalg.eigvalsh((A + A.T) / 2))
    poisson = LevelSpacing.ratio_statistic(rng.uniform(0, 1, 600))
    assert goe["r_mean"] == pytest.approx(0.5307, abs=0.03)
    assert poisson["r_mean"] == pytest.approx(0.3863, abs=0.03)
    assert goe["diagnosis"].startswith("GOE")
    assert poisson["diagnosis"].startswith("Poisson")


def test_spacing_distributions_are_normalized():
    s = np.linspace(0, 20, 20001)
    for density in (LevelSpacing.wigner_surmise(s), LevelSpacing.poisson_distribution(s),
                    LevelSpacing.gue_surmise(s), LevelSpacing.brody_distribution(s, 0.4)):
        assert trapezoid(density, s) == pytest.approx(1.0, abs=1e-3)


def test_otoc_starts_at_zero_for_commuting_operators():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4))
    B = rng.normal(size=(4, 4))
    W = np.kron(A + A.T, np.eye(4))
    V = np.kron(np.eye(4), B + B.T)
    H = rng.normal(size=(16, 16))
    H = (H + H.T) / 2
    otoc = LyapunovEstimator().compute_otoc(H, W, V, np.array([0.0, 0.5, 1.0]))
    assert otoc[0] == pytest.approx(0.0, abs=1e-10)
    assert otoc[2] > otoc[0]


def test_husimi_matches_coherent_state_overlaps():
    top = KickedTop(j=6, k=3.0)
    state = top.evolve(top.coherent_state(1.0, 0.5), 7)
    thetas, phis = np.linspace(0, np.pi, 5), np.linspace(0, 2 * np.pi, 7)
    Q = top.husimi_q(state, thetas, phis)
    prefactor = (2 * top.j + 1) / (4 * np.pi)
    for i, theta in enumerate(thetas):
        for k, phi in enumerate(phis):
            expected = prefactor * abs(np.vdot(top.coherent_state(theta, phi), state)) ** 2
            assert Q[i, k] == pytest.approx(expected, abs=1e-12)
