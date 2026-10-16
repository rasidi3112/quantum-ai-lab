from __future__ import annotations

import numpy as np
import pytest

from quantum_rl import CartPoleEnv, FrozenLakeEnv, HybridAgent, QuantumPolicy


def test_cartpole_step_contract():
    env = CartPoleEnv(max_steps=5, seed=0)
    obs = env.reset()
    assert obs.shape == (4,)
    total, done = 0.0, False
    while not done:
        obs, reward, done, info = env.step(1)
        total += reward
    assert total <= 5
    assert "theta_deg" in info


def test_frozen_lake_deterministic_path_reaches_goal():
    env = FrozenLakeEnv(is_slippery=False, seed=0)
    env.reset()
    for action in (2, 2, 1, 1, 2, 1):
        obs, reward, done, _ = env.step(action)
    assert done and reward == 1.0
    assert obs.argmax() == env.n_states - 1


def test_policy_outputs_distribution(rng):
    policy = QuantumPolicy(n_qubits=4, n_layers=2, n_actions=3, seed=0)
    probs = policy.forward(rng.normal(size=4), policy.init_params())
    assert probs.shape == (3,)
    assert probs.sum() == pytest.approx(1.0)


def test_policy_cnot_matches_permutation(rng):
    policy = QuantumPolicy(n_qubits=3, seed=0)
    state = rng.normal(size=8) + 1j * rng.normal(size=8)
    original = state.copy()
    out = policy._apply_cnot(state, control=0, target=2)
    expected = original.copy()
    for k in range(8):
        if k & 0b100:
            expected[k ^ 0b001] = original[k]
    assert np.allclose(out, expected)
    assert np.allclose(state, original)


def test_agent_updates_quantum_and_classical_parameters():
    env = CartPoleEnv(seed=0)
    agent = HybridAgent(seed=0, input_scale=env.obs_scale)
    W0, q0 = agent.classical_W.copy(), agent.quantum_params.copy()
    agent.train(env, episodes=2, verbose=False)
    assert not np.allclose(agent.classical_W, W0)
    assert not np.allclose(agent.quantum_params, q0)


def test_agent_save_load_roundtrip(tmp_path):
    env = CartPoleEnv(seed=0)
    agent = HybridAgent(seed=1, input_scale=env.obs_scale)
    path = tmp_path / "agent.json"
    agent.save_params(str(path))
    clone = HybridAgent(seed=99)
    clone.load_params(str(path))
    obs = env.reset()
    assert np.allclose(agent.get_action_probs(obs), clone.get_action_probs(obs))
