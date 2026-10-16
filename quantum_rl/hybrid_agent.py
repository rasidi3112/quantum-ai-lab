"""Hybrid classical–quantum REINFORCE agent.

A linear layer with ``pi * tanh`` squashing maps (scaled) observations to
rotation angles; a :class:`~quantum_rl.quantum_policy.QuantumPolicy` turns them
into action probabilities. Both parts are trained jointly with the
REINFORCE policy gradient and a running-mean baseline.

References
----------
Williams, R. J. "Simple statistical gradient-following algorithms for
connectionist reinforcement learning." Mach. Learn. 8, 229 (1992).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from quantum_rl.quantum_policy import QuantumPolicy


@dataclass
class TrainingResult:
    """History of a training run.

    Attributes
    ----------
    episode_rewards : list of float
        Total reward of each episode.
    episode_lengths : list of int
        Length of each episode.
    best_reward : float
        Highest episode reward.
    best_episode : int
        Index of that episode.
    final_params : numpy.ndarray or None
        Quantum parameters after training.
    final_classical_params : dict or None
        Classical weights ``W`` and bias ``b`` after training.
    """

    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    best_reward: float = float("-inf")
    best_episode: int = -1
    final_params: np.ndarray | None = None
    final_classical_params: dict[str, np.ndarray] | None = None


class HybridAgent:
    """Policy-gradient agent with a classical encoder and a quantum policy.

    Parameters
    ----------
    n_qubits : int, default 4
        Qubits of the quantum policy.
    n_layers : int, default 2
        Variational layers.
    n_actions : int, default 2
        Number of discrete actions.
    obs_dim : int, default 4
        Observation dimension.
    gamma : float, default 0.99
        Discount factor.
    seed : int, optional
        Seed for initialization and action sampling.
    input_scale : numpy.ndarray, optional
        Observations are divided by this before the linear layer, e.g.
        :attr:`CartPoleEnv.obs_scale`.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        n_actions: int = 2,
        obs_dim: int = 4,
        gamma: float = 0.99,
        seed: int | None = None,
        input_scale: np.ndarray | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.input_scale = (
            np.ones(obs_dim) if input_scale is None else np.asarray(input_scale, dtype=float)
        )

        self.policy = QuantumPolicy(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_actions=n_actions,
            seed=seed,
        )

        limit = np.sqrt(6.0 / (obs_dim + n_qubits))
        self.classical_W = self.rng.uniform(-limit, limit, size=(n_qubits, obs_dim))
        self.classical_b = np.zeros(n_qubits)

        n_params = QuantumPolicy.n_params(n_qubits, n_layers)
        self.quantum_params = self.rng.normal(0, 0.1, size=n_params)

        self._baseline = 0.0
        self._baseline_count = 0


    def _preprocess(self, observation: np.ndarray) -> np.ndarray:
        return self._preprocess_batch(np.asarray(observation)[None, :])[0]

    def _preprocess_batch(self, observations: np.ndarray) -> np.ndarray:
        z = (observations / self.input_scale) @ self.classical_W.T + self.classical_b
        return np.pi * np.tanh(z)

    def get_action_probs(
        self, observation: np.ndarray
    ) -> np.ndarray:
        """Return the action probabilities ``pi(. | observation)``."""
        encoded = self._preprocess(observation)
        return self.policy.forward(encoded, self.quantum_params)

    def select_action(self, observation: np.ndarray) -> int:
        """Sample an action from the current policy."""
        probs = self.get_action_probs(observation)
        return int(self.rng.choice(self.n_actions, p=probs))


    @staticmethod
    def _compute_returns(
        rewards: list[float], gamma: float
    ) -> np.ndarray:
        T = len(rewards)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G
        return returns

    def _collect_episode(self, env: Any) -> tuple[
        list[np.ndarray],
        list[int],
        list[float],
        int,
    ]:
        observations: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []

        obs = env.reset()
        done = False

        while not done:
            observations.append(obs.copy())
            action = self.select_action(obs)
            actions.append(action)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)

        return observations, actions, rewards, len(rewards)


    def _update_baseline(self, total_return: float) -> float:
        baseline = self._baseline
        self._baseline_count += 1
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
        """Train with REINFORCE.

        Parameters
        ----------
        env : object
            Environment with ``reset()`` and ``step(action)`` methods.
        episodes : int, default 200
            Number of training episodes.
        lr_quantum : float, default 0.01
            Learning rate of the circuit parameters.
        lr_classical : float, default 0.001
            Learning rate of the linear layer.
        epsilon : float, default 1e-3
            Finite-difference step for the gradients.
        verbose : bool, default True
            Print progress.
        print_every : int, default 20
            Episodes between progress lines.

        Returns
        -------
        TrainingResult
        """
        result = TrainingResult()

        if verbose:
            print(f"Training HybridAgent: {self.n_qubits}q × {self.n_layers}L, "
                  f"{len(self.quantum_params)} quantum params, "
                  f"{self.classical_W.size + self.classical_b.size} classical params")
            print(f"  Episodes: {episodes}, γ={self.gamma}, "
                  f"lr_q={lr_quantum}, lr_c={lr_classical}")

        for ep in range(episodes):
            observations, actions, rewards, ep_len = self._collect_episode(env)
            total_reward = sum(rewards)

            result.episode_rewards.append(total_reward)
            result.episode_lengths.append(ep_len)

            if total_reward > result.best_reward:
                result.best_reward = total_reward
                result.best_episode = ep

            returns = self._compute_returns(rewards, self.gamma)
            baseline = self._update_baseline(total_reward)
            advantages = returns - baseline

            if len(advantages) > 1:
                adv_std = np.std(advantages)
                if adv_std > 1e-8:
                    advantages = (advantages - np.mean(advantages)) / adv_std

            obs_batch = np.array(observations)
            action_batch = np.array(actions)
            q_grad = self.policy.batch_policy_gradient(
                self._preprocess_batch(obs_batch), action_batch, advantages,
                self.quantum_params, epsilon=epsilon,
            )
            self.quantum_params += lr_quantum * q_grad

            c_grad_W = np.zeros_like(self.classical_W)
            c_grad_b = np.zeros_like(self.classical_b)

            for i in range(self.n_qubits):
                for j in range(self.obs_dim):
                    self.classical_W[i, j] += epsilon
                    j_plus = self._evaluate_episode_log_prob(
                        obs_batch, action_batch, advantages
                    )
                    self.classical_W[i, j] -= 2 * epsilon
                    j_minus = self._evaluate_episode_log_prob(
                        obs_batch, action_batch, advantages
                    )
                    self.classical_W[i, j] += epsilon
                    c_grad_W[i, j] = (j_plus - j_minus) / (2 * epsilon)

                self.classical_b[i] += epsilon
                j_plus = self._evaluate_episode_log_prob(
                    obs_batch, action_batch, advantages
                )
                self.classical_b[i] -= 2 * epsilon
                j_minus = self._evaluate_episode_log_prob(
                    obs_batch, action_batch, advantages
                )
                self.classical_b[i] += epsilon
                c_grad_b[i] = (j_plus - j_minus) / (2 * epsilon)

            self.classical_W += lr_classical * c_grad_W
            self.classical_b += lr_classical * c_grad_b

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
        observations: np.ndarray,
        actions: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        if weights is None:
            weights = np.ones(len(actions))
        probs = self.policy.forward_batch(
            self._preprocess_batch(np.asarray(observations)), self.quantum_params
        )
        log_probs = np.log(probs[np.arange(len(actions)), np.asarray(actions)] + 1e-15)
        return float(np.mean(weights * log_probs))


    def evaluate(
        self, env: Any, n_episodes: int = 10, verbose: bool = False
    ) -> dict[str, float]:
        """Run greedy (argmax) episodes without learning.

        Parameters
        ----------
        env : object
            Environment with ``reset()`` and ``step(action)`` methods.
        n_episodes : int, default 10
            Number of episodes.
        verbose : bool, default False
            Print each episode.

        Returns
        -------
        dict
            Mean, standard deviation, maximum and minimum reward, and mean length.
        """
        rewards = []
        lengths = []

        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
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


    def save_params(self, filepath: str) -> None:
        """Save all parameters to a JSON file."""
        data = {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_actions": self.n_actions,
            "obs_dim": self.obs_dim,
            "gamma": self.gamma,
            "input_scale": self.input_scale.tolist(),
            "quantum_params": self.quantum_params.tolist(),
            "classical_W": self.classical_W.tolist(),
            "classical_b": self.classical_b.tolist(),
            "baseline": self._baseline,
            "baseline_count": self._baseline_count,
        }
        Path(filepath).write_text(json.dumps(data, indent=2))

    def load_params(self, filepath: str) -> None:
        """Load parameters saved with :meth:`save_params`."""
        data = json.loads(Path(filepath).read_text())
        self.quantum_params = np.array(data["quantum_params"])
        self.classical_W = np.array(data["classical_W"])
        self.classical_b = np.array(data["classical_b"])
        self.input_scale = np.array(data.get("input_scale", np.ones(self.obs_dim)))
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
