"""Self-contained reinforcement-learning environments.

Gym-style ``reset()`` / ``step(action)`` environments with no external
dependencies: the classic CartPole balancing task and the FrozenLake grid
world.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class CartPoleEnv:
    """Cart-pole balancing task with the classic Gym dynamics.

    The agent pushes the cart left (0) or right (1) and receives +1 per step
    until the pole tilts beyond 12 degrees, the cart leaves the track, or
    ``max_steps`` is reached.

    Parameters
    ----------
    max_steps : int, default 200
        Episode length limit.
    seed : int, optional
        Seed for the initial state.

    Attributes
    ----------
    obs_scale : numpy.ndarray
        Typical magnitude of each observation component, useful for scaling
        inputs to a quantum policy.
    """

    def __init__(self, max_steps: int = 200, seed: int | None = None) -> None:
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.pole_half_length = 0.5
        self.force_mag = 10.0
        self.dt = 0.02

        self.theta_threshold = 12 * np.pi / 180
        self.x_threshold = 2.4

        self.n_actions = 2
        self.obs_dim = 4
        self.obs_scale = np.array([self.x_threshold, 2.0, self.theta_threshold, 2.0])
        self.name = "CartPole-v0"

        self.state: np.ndarray | None = None
        self.steps: int = 0

    def reset(self) -> np.ndarray:
        """Start a new episode.

        Returns
        -------
        numpy.ndarray
            Observation ``[x, x_dot, theta, theta_dot]``.
        """
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance the simulation by one time step.

        Parameters
        ----------
        action : {0, 1}
            Push left or right.

        Returns
        -------
        observation : numpy.ndarray
        reward : float
        done : bool
        info : dict
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step()")

        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag if action == 1 else -self.force_mag

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        temp = (
            force + self.mass_pole * self.pole_half_length * theta_dot ** 2 * sin_theta
        ) / self.total_mass

        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.pole_half_length
            * (4.0 / 3.0 - self.mass_pole * cos_theta ** 2 / self.total_mass)
        )

        x_acc = temp - self.mass_pole * self.pole_half_length * theta_acc * cos_theta / self.total_mass

        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        done = bool(
            abs(x) > self.x_threshold
            or abs(theta) > self.theta_threshold
            or self.steps >= self.max_steps
        )

        reward = 1.0

        info = {
            "steps": self.steps,
            "x": x,
            "theta_deg": np.degrees(theta),
        }

        return self.state.copy(), reward, done, info

    def render_ascii(self) -> str:
        """Return a text drawing of the cart and pole."""
        if self.state is None:
            return "Environment not initialized. Call reset()."

        x, _, theta, _ = self.state
        width = 40
        cart_pos = int((x / self.x_threshold + 1) * width / 2)
        cart_pos = max(2, min(width - 3, cart_pos))

        if abs(theta) < 0.05:
            pole_char = "|"
        elif theta > 0:
            pole_char = "/"
        else:
            pole_char = "\\"

        lines = []
        lines.append("=" * (width + 2))

        pole_line = [" "] * (width + 2)
        pole_line[cart_pos + 1] = pole_char
        lines.append("".join(pole_line))

        cart_line = [" "] * (width + 2)
        cart_line[cart_pos] = "["
        cart_line[cart_pos + 1] = "█"
        cart_line[cart_pos + 2] = "]"
        lines.append("".join(cart_line))

        lines.append("-" * (width + 2))
        lines.append(
            f" x={x:+.2f}  θ={np.degrees(theta):+.1f}°  step={self.steps}"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CartPoleEnv(max_steps={self.max_steps})"


class FrozenLakeEnv:
    """FrozenLake grid world: reach the goal without falling into a hole.

    Parameters
    ----------
    is_slippery : bool, default True
        With probability 2/3 the agent moves perpendicular to the chosen action.
    max_steps : int, default 100
        Episode length limit.
    custom_map : list of str, optional
        Map rows using ``S`` (start), ``F`` (frozen), ``H`` (hole) and ``G`` (goal).
    seed : int, optional
        Seed for slippery transitions.
    """

    DEFAULT_MAP = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ]

    ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    ACTION_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    def __init__(
        self,
        is_slippery: bool = True,
        max_steps: int = 100,
        custom_map: list | None = None,
        seed: int | None = None,
    ) -> None:
        self.is_slippery = is_slippery
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.grid_map = custom_map if custom_map else self.DEFAULT_MAP
        self.nrow = len(self.grid_map)
        self.ncol = len(self.grid_map[0])

        self.n_actions = 4
        self.n_states = self.nrow * self.ncol
        self.obs_dim = self.n_states

        self.name = "FrozenLake-v0"

        self.start_pos = None
        for r, row in enumerate(self.grid_map):
            for c, cell in enumerate(row):
                if cell == "S":
                    self.start_pos = (r, c)
        if self.start_pos is None:
            self.start_pos = (0, 0)

        self.agent_pos: tuple[int, int] | None = None
        self.steps: int = 0

    def _pos_to_state(self, pos: tuple[int, int]) -> int:
        return pos[0] * self.ncol + pos[1]

    def _state_to_pos(self, state: int) -> tuple[int, int]:
        return (state // self.ncol, state % self.ncol)

    def _get_cell(self, pos: tuple[int, int]) -> str:
        return self.grid_map[pos[0]][pos[1]]

    def _move(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        r, c = pos
        if action == 0:
            r = max(0, r - 1)
        elif action == 1:
            c = min(self.ncol - 1, c + 1)
        elif action == 2:
            r = min(self.nrow - 1, r + 1)
        elif action == 3:
            c = max(0, c - 1)
        return (r, c)

    def reset(self) -> np.ndarray:
        """Start a new episode.

        Returns
        -------
        numpy.ndarray
            One-hot encoding of the agent's cell.
        """
        self.agent_pos = self.start_pos
        self.steps = 0
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        if self.agent_pos is None:
            raise RuntimeError("Call reset() before reading observations")
        obs = np.zeros(self.n_states, dtype=np.float64)
        obs[self._pos_to_state(self.agent_pos)] = 1.0
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Move the agent.

        Parameters
        ----------
        action : {0, 1, 2, 3}
            Up, right, down or left.

        Returns
        -------
        observation : numpy.ndarray
        reward : float
            1 on reaching the goal, else 0.
        done : bool
        info : dict
        """
        if self.agent_pos is None:
            raise RuntimeError("Call reset() before step()")

        if self.is_slippery:
            rand = self.rng.random()
            if rand < 1 / 3:
                actual_action = action
            elif rand < 2 / 3:
                actual_action = (action + 1) % 4
            else:
                actual_action = (action - 1) % 4
        else:
            actual_action = action

        self.agent_pos = self._move(self.agent_pos, actual_action)
        self.steps += 1

        cell = self._get_cell(self.agent_pos)

        if cell == "H":
            reward = 0.0
            done = True
        elif cell == "G":
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = self.steps >= self.max_steps

        info = {
            "steps": self.steps,
            "position": self.agent_pos,
            "cell": cell,
            "actual_action": self.ACTION_NAMES.get(actual_action, "?"),
        }

        return self._get_observation(), reward, done, info

    def render_ascii(self) -> str:
        """Return a text drawing of the grid and the agent."""
        if self.agent_pos is None:
            return "Environment not initialized. Call reset()."

        lines = []
        lines.append("+" + "---+" * self.ncol)

        for r in range(self.nrow):
            row_str = "|"
            for c in range(self.ncol):
                cell = self.grid_map[r][c]
                if (r, c) == self.agent_pos:
                    row_str += " A |"
                elif cell == "S":
                    row_str += " S |"
                elif cell == "G":
                    row_str += " G |"
                elif cell == "H":
                    row_str += " ■ |"
                else:
                    row_str += "   |"
            lines.append(row_str)
            lines.append("+" + "---+" * self.ncol)

        lines.append(f"Step: {self.steps}  Pos: {self.agent_pos}")
        lines.append(f"Slippery: {self.is_slippery}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"FrozenLakeEnv(slippery={self.is_slippery}, max_steps={self.max_steps})"
