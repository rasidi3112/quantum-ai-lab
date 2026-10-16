from __future__ import annotations

import numpy as np
import pytest

from quantum_simulation import HubbardModel, IsingModel, TrotterEvolution
from quantum_simulation.ising_model import identity, sigma_x, sigma_z, tensor_product


def exact_tfim_energy(n: int, J: float, h: float) -> float:
    k = np.pi * (2 * np.arange(n) + 1) / n
    return -np.sum(np.sqrt(J ** 2 + h ** 2 - 2 * J * h * np.cos(k)))


@pytest.mark.parametrize("n, h", [(4, 0.3), (6, 0.5), (8, 1.0), (7, 1.7)])
def test_ising_matches_free_fermion_solution(n, h):
    E0, psi0 = IsingModel(n, J=1.0, h=h).get_ground_state()
    assert E0 == pytest.approx(exact_tfim_energy(n, 1.0, h), abs=1e-9)
    assert np.linalg.norm(psi0) == pytest.approx(1.0)


def test_ising_hamiltonian_is_hermitian():
    H = IsingModel(5, 0.7, 1.3).hamiltonian
    assert np.allclose(H, H.conj().T)


def test_ising_long_range_order_vanishes_in_paramagnet():
    def corr(h):
        _, psi = IsingModel(8, 1.0, h).get_ground_state()
        return IsingModel.correlation_function(psi, 0, 4, 8)
    assert corr(0.2) > 0.95
    assert corr(2.0) < 0.1


def test_time_evolve_handles_non_divisible_steps():
    model = IsingModel(4, 1.0, 0.7)
    psi = np.zeros(16, dtype=complex)
    psi[0] = 1.0
    approx = IsingModel.time_evolve(psi, model.hamiltonian, t=1.0, dt=0.3)
    exact = TrotterEvolution.exact_evolution(psi, model.hamiltonian, 1.0)
    assert np.allclose(approx, exact)


def _ising_terms(n: int, h: float):
    zz = sum(
        tensor_product(*[sigma_z if k in (i, (i + 1) % n) else identity for k in range(n)])
        for i in range(n)
    )
    x = sum(tensor_product(*[sigma_x if k == i else identity for k in range(n)]) for i in range(n))
    return [-zz, -h * x]


@pytest.mark.parametrize("order, expected_ratio", [(1, 4), (2, 16), (4, 256)])
def test_trotter_error_scales_with_order(order, expected_ratio):
    terms = _ising_terms(4, 0.8)
    psi = np.zeros(16, dtype=complex)
    psi[0] = 1.0
    errors = TrotterEvolution.compare_orders(psi, terms, 1.0, [0.1, 0.05])[order]
    assert errors[0] / errors[1] == pytest.approx(expected_ratio, rel=0.25)


def test_trotter_rejects_unknown_order():
    with pytest.raises(ValueError):
        TrotterEvolution.evolve(np.ones(4) / 2, _ising_terms(2, 1.0), 1.0, 0.1, order=3)


@pytest.mark.parametrize("U", [0.0, 2.0, 8.0])
def test_hubbard_dimer_half_filling_energy(U):
    model = HubbardModel(n_sites=2, t_hop=1.0, U=U)
    H = model.hamiltonian
    assert np.allclose(H, H.conj().T)
    half = np.array([bin(i).count("1") == 2 for i in range(model.dim)])
    E0 = np.linalg.eigvalsh(H[np.ix_(half, half)])[0]
    assert E0 == pytest.approx(U / 2 - np.sqrt(U ** 2 / 4 + 4), abs=1e-10)


def test_hubbard_density_counts_particles():
    model = HubbardModel(n_sites=2, t_hop=1.0, U=4.0)
    half = np.array([bin(i).count("1") == 2 for i in range(model.dim)])
    _, vecs = np.linalg.eigh(model.hamiltonian[np.ix_(half, half)])
    psi = np.zeros(model.dim, dtype=complex)
    psi[half] = vecs[:, 0]
    assert model.density_profile(psi).sum() == pytest.approx(2.0)
    assert np.all(model.double_occupancy(psi) < 0.25)
