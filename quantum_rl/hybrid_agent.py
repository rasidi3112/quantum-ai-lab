"""
Hybrid Classical-Quantum Reinforcement Learning Agent
======================================================

Implements a hybrid RL agent that combines a classical preprocessing layer
with a variational quantum circuit (VQC) for policy decisions.  Training
uses the REINFORCE (Monte-Carlo policy gradient) algorithm.

Architecture::

    Classical                   Quantum                    Classical
    ┌──────────┐   angles   ┌──────────────┐   probs   ┌──────────┐
    │  Linear  │ ────────→ │ QuantumPolicy │ ───────→ │ Softmax  │ → action
    │ W·obs+b  │           │  VQC circuit  │           │          │
    └──────────┘           └──────────────┘           └──────────┘

The classical linear layer maps raw observations to rotation angles
(dimensionality reduction if obs_dim > n_qubits).  The quantum policy
circuit produces action probabilities, which are normalised via softmax.

Training procedure (REINFORCE with baseline):
    1. Collect full episode trajectory τ = {(s_t, a_t, r_t)}
    2. Compute discounted returns G_t = Σ_{k=0}^{T-t} γ^k · r_{t+k}
    3. Compute advantage: A_t = G_t − baseline  (baseline = running mean)
    4. Update quantum params via finite-difference policy gradient
    5. Update classical params via vanilla SGD

References
----------
- Jerbi et al., "Parametrized quantum policies for reinforcement learning" (2021)
- Skolik et al., "Quantum agents in the Gym" (2022)
- Lockwood & Si, "Reinforcement Learning with Quantum Variational Circuits" (2020)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from quantum_rl.quantum_policy import QuantumPolicy


# ---------------------------------------------------------------------------
# Training result container
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Container for training diagnostics."""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    best_reward: float = float("-inf")
    best_episode: int = -1
    final_params: Optional[np.ndarray] = None
    final_classical_params: Optional[Dict[str, np.ndarray]] = None


# ---------------------------------------------------------------------------
# Hybrid Agent
# ---------------------------------------------------------------------------

class HybridAgent:
    """Hybrid classical-quantum reinforcement learning agent.

    The agent uses a classical linear layer for observation preprocessing
    followed by a QuantumPolicy circuit for action selection.  Training
    uses the REINFORCE algorithm with a moving-average baseline for
    variance reduction.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the quantum policy circuit.
    n_layers : int
        Number of variational layers in the circuit.
    n_actions : int
        Number of discrete actions.
    obs_dim : int
        Dimensionality of the observation vector.
    gamma : float
        Discount factor for returns.
    seed : int | None
        Random seed for reproducibility.

    Attributes
    ----------
    quantum_params : np.ndarray
        Trainable parameters of the variational quantum circuit.
    classical_W : np.ndarray
        Weight matrix of the classical linear layer (n_qubits × obs_dim).
    classical_b : np.ndarray
        Bias vector of the classical linear layer (n_qubits,).
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        n_actions: int = 2,
        obs_dim: int = 4,
        gamma: float = 0.99,
        seed: Optional[int] = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.gamma = gamma
        self._rng = np.random.default_rng(seed)

        # Quantum policy
        self.policy = QuantumPolicy(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_actions=n_actions,
            seed=seed,
        )

        # Classical pre-processing layer: obs → rotation angles
        # Xavier-like initialisation
        limit = np.sqrt(6.0 / (obs_dim + n_qubits))
        self.classical_W = self._rng.uniform(-limit, limit, size=(n_qubits, obs_dim))
        self.classical_b = np.zeros(n_qubits)

        # Quantum circuit parameters
        n_params = QuantumPolicy.n_params(n_qubits, n_layers)
        self.quantum_params = self._rng.normal(0, 0.1, size=n_params)

        # Baseline (moving average of returns) for variance reduction
        self._baseline = 0.0
        self._baseline_count = 0

    # ----- forward pass -------------------------------------------------- #

    def _preprocess(self, observation: np.ndarray) -> np.ndarray:
        """Classical linear layer: obs → encoded angles.

        Parameters
        ----------
        observation : np.ndarray, shape ``(obs_dim,)``

        Returns
        -------
        np.ndarray, shape ``(n_qubits,)``
            Encoded angles to feed into the quantum circuit.
        """
        return np.tanh(self.classical_W @ observation + self.classical_b)

    def get_action_probs(
        self, observation: np.ndarray
    ) -> np.ndarray:
        """Compute action probability distribution.

        Parameters
        ----------
        observation : np.ndarray

        Returns
        -------
        np.ndarray, shape ``(n_actions,)``
        """
        encoded = self._preprocess(observation)
        return self.policy.forward(encoded, self.quantum_params)

    def select_action(self, observation: np.ndarray) -> int:
        """Sample an action from the current policy.

        Parameters
        ----------
        observation : np.ndarray

        Returns
        -------
        int
            Selected action index.
        """
        probs = self.get_action_probs(observation)
        return int(self._rng.choice(self.n_actions, p=probs))

    # ----- trajectory collection ----------------------------------------- #

    @staticmethod
    def _compute_returns(
        rewards: List[float], gamma: float
    ) -> np.ndarray:
        """Compute discounted returns G_t = Σ γ^k r_{t+k}.

        Parameters
        ----------
        rewards : list of float
            Episode rewards [r_0, r_1, ..., r_T].
        gamma : float
            Discount factor.

        Returns
        -------
        np.ndarray
            Discounted returns aligned with each timestep.
        """
        T = len(rewards)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G
        return returns

    def _collect_episode(self, env: Any) -> Tuple[
        List[np.ndarray],  # observations
        List[int],         # actions
        List[float],       # rewards
        int,               # episode length
    ]:
        """Run a single episode and collect the trajectory.

        Parameters
        ----------
        env
            Environment with ``reset()`` and ``step(action)`` methods.

        Returns
        -------
        tuple
            (observations, actions, rewards, episode_length)
        """
        observations: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[float] = []

        obs = env.reset()
        done = False

        while not done:
            observations.append(obs.copy())
            action = self.select_action(obs)
            actions.append(action)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)

        return observations, actions, rewards, len(rewards)

    # ----- training ------------------------------------------------------ #

    def _update_baseline(self, total_return: float) -> float:
        """Update running-average baseline and return the current one.

        Parameters
        ----------
        total_return : float
            Undiscounted total reward of the episode.

        Returns
        -------
        float
            Current baseline value (before update).
        """
        baseline = self._baseline
        self._baseline_count += 1
        # Incremental mean update
        self._baseline += (total_return - self._baseline) / self._baseline_count
        return baseline

    def train(
        self,
        env: Any,
        episodes: int = 200,
        lr_quantum: float = 0.01,
        lr_classical: float = 0.001,
        epsilon: float = 1e-3,
        verbose: bool = True,
        print_every: int = 20,
    ) -> TrainingResult:
        """Train the hybrid agent using REINFORCE.

        Parameters
        ----------
        env
            RL environment with ``reset()`` and ``step(action)`` methods.
        episodes : int
            Number of training episodes.
        lr_quantum : float
            Learning rate for quantum circuit parameters.
        lr_classical : float
            Learning rate for classical layer parameters.
        epsilon : float
            Finite-difference step size for policy gradient.
        verbose : bool
            Print training progress.
        print_every : int
            Print frequency (episodes).

        Returns
        -------
        TrainingResult
            Training diagnostics including reward history.
        """
        result = TrainingResult()

        if verbose:
            print(f"Training HybridAgent: {self.n_qubits}q × {self.n_layers}L, "
                  f"{len(self.quantum_params)} quantum params, "
                  f"{self.classical_W.size + self.classical_b.size} classical params")
            print(f"  Episodes: {episodes}, γ={self.gamma}, "
                  f"lr_q={lr_quantum}, lr_c={lr_classical}")

        for ep in range(episodes):
            # Collect trajectory
            observations, actions, rewards, ep_len = self._collect_episode(env)
            total_reward = sum(rewards)

            result.episode_rewards.append(total_reward)
            result.episode_lengths.append(ep_len)

            if total_reward > result.best_reward:
                result.best_reward = total_reward
                result.best_episode = ep

            # Compute discounted returns and advantage
            returns = self._compute_returns(rewards, self.gamma)
            baseline = self._update_baseline(total_reward)
            advantages = returns - baseline

            # Normalise advantages
            if len(advantages) > 1:
                adv_std = np.std(advantages)
                if adv_std > 1e-8:
                    advantages = (advantages - np.mean(advantages)) / adv_std

            # --- Policy gradient for quantum parameters ---
            q_grad = np.zeros_like(self.quantum_params)
            for t in range(len(observations)):
                encoded = self._preprocess(observations[t])
                step_grad = self.policy.policy_gradient(
                    encoded, actions[t], advantages[t],
                    self.quantum_params, epsilon=epsilon,
                )
                q_grad += step_grad

            # Average gradient over timesteps
            q_grad /= len(observations)

            # Update quantum parameters (gradient ascent on expected return)
            self.quantum_params += lr_quantum * q_grad

            # --- Classical parameter update (finite-difference SGD) ---
            c_grad_W = np.zeros_like(self.classical_W)
            c_grad_b = np.zeros_like(self.classical_b)

            for i in range(self.n_qubits):
                for j in range(self.obs_dim):
                    # Perturb W[i, j]
                    self.classical_W[i, j] += epsilon
                    probs_plus = self._evaluate_episode_log_prob(
                        observations, actions
                    )
                    self.classical_W[i, j] -= 2 * epsilon
                    probs_minus = self._evaluate_episode_log_prob(
                        observations, actions
                    )
                    self.classical_W[i, j] += epsilon  # restore

                    grad = (probs_plus - probs_minus) / (2 * epsilon)
                    c_grad_W[i, j] = grad * np.mean(advantages)

                # Perturb b[i]
                self.classical_b[i] += epsilon
                probs_plus = self._evaluate_episode_log_prob(
                    observations, actions
                )
                self.classical_b[i] -= 2 * epsilon
                probs_minus = self._evaluate_episode_log_prob(
                    observations, actions
                )
                self.classical_b[i] += epsilon  # restore

                grad = (probs_plus - probs_minus) / (2 * epsilon)
                c_grad_b[i] = grad * np.mean(advantages)

            self.classical_W += lr_classical * c_grad_W
            self.classical_b += lr_classical * c_grad_b

            # Logging
            if verbose and (ep + 1) % print_every == 0:
                recent = result.episode_rewards[-print_every:]
                avg = np.mean(recent)
                print(f"  Episode {ep+1:4d}/{episodes} | "
                      f"Reward: {total_reward:6.1f} | "
                      f"Avg({print_every}): {avg:6.1f} | "
                      f"Best: {result.best_reward:.1f}")

        result.final_params = self.quantum_params.copy()
        result.final_classical_params = {
            "W": self.classical_W.copy(),
            "b": self.classical_b.copy(),
        }

        if verbose:
            print(f"  Training complete. Best reward: {result.best_reward:.1f} "
                  f"(episode {result.best_episode + 1})")

        return result

    def _evaluate_episode_log_prob(
        self,
        observations: List[np.ndarray],
        actions: List[int],
    ) -> float:
        """Compute mean log-probability of actions under current policy.

        Parameters
        ----------
        observations : list of np.ndarray
        actions : list of int

        Returns
        -------
        float
            Mean log π(a_t | s_t).
        """
        total_log_prob = 0.0
        for obs, act in zip(observations, actions):
            probs = self.get_action_probs(obs)
            total_log_prob += np.log(probs[act] + 1e-15)
        return total_log_prob / len(observations)

    # ----- evaluation ---------------------------------------------------- #

    def evaluate(
        self, env: Any, n_episodes: int = 10, verbose: bool = False
    ) -> Dict[str, float]:
        """Evaluate the agent's performance (greedy policy).

        Parameters
        ----------
        env
            Environment to evaluate in.
        n_episodes : int
            Number of evaluation episodes.
        verbose : bool
            Print per-episode results.

        Returns
        -------
        dict
            Statistics including mean_reward, std_reward, mean_length.
        """
        rewards = []
        lengths = []

        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                # Greedy: pick action with highest probability
                probs = self.get_action_probs(obs)
                action = int(np.argmax(probs))
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                steps += 1

            rewards.append(total_reward)
            lengths.append(steps)

            if verbose:
                print(f"  Eval episode {ep+1}: reward={total_reward:.1f}, "
                      f"length={steps}")

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "mean_length": float(np.mean(lengths)),
        }

    # ----- serialisation ------------------------------------------------- #

    def save_params(self, filepath: str) -> None:
        """Save agent parameters to a JSON file.

        Parameters
        ----------
        filepath : str
            Path to save the parameter file.
        """
        data = {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_actions": self.n_actions,
            "obs_dim": self.obs_dim,
            "gamma": self.gamma,
            "quantum_params": self.quantum_params.tolist(),
            "classical_W": self.classical_W.tolist(),
            "classical_b": self.classical_b.tolist(),
            "baseline": self._baseline,
            "baseline_count": self._baseline_count,
        }
        Path(filepath).write_text(json.dumps(data, indent=2))

    def load_params(self, filepath: str) -> None:
        """Load agent parameters from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the saved parameter file.
        """
        data = json.loads(Path(filepath).read_text())
        self.quantum_params = np.array(data["quantum_params"])
        self.classical_W = np.array(data["classical_W"])
        self.classical_b = np.array(data["classical_b"])
        self._baseline = data.get("baseline", 0.0)
        self._baseline_count = data.get("baseline_count", 0)

    def __repr__(self) -> str:
        n_q = len(self.quantum_params)
        n_c = self.classical_W.size + self.classical_b.size
        return (
            f"HybridAgent(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"n_actions={self.n_actions}, obs_dim={self.obs_dim}, "
            f"quantum_params={n_q}, classical_params={n_c})"
        )
