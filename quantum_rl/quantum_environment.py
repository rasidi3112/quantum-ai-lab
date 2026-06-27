"""
Quantum Environments for Reinforcement Learning
=================================================

Built-in environments with a common interface, requiring no external
dependencies (no OpenAI Gym needed). Each environment follows the
standard RL interface:

    state = env.reset()
    state, reward, done, info = env.step(action)

Environments:
    - CartPoleEnv: Classic inverted pendulum control task
    - FrozenLakeEnv: Grid world navigation with slippery ice

The CartPole physics use Euler integration of the equations of motion
for a cart-pole system. The FrozenLake uses a 4×4 grid world with
optional stochastic transitions.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


class CartPoleEnv:
    """
    Simplified CartPole environment (no OpenAI Gym dependency).

    A pole is attached by an un-actuated joint to a cart that moves along
    a frictionless track. The system is controlled by applying a force of
    +1 or -1 to the cart. The goal is to keep the pole upright.

    State space: [x, x_dot, theta, theta_dot]
        x       : cart position
        x_dot   : cart velocity
        theta   : pole angle (radians, 0 = upright)
        theta_dot : pole angular velocity

    Action space: {0, 1}
        0 : push left  (force = -10 N)
        1 : push right (force = +10 N)

    Episode termination:
        - Pole angle |θ| > 12° (≈0.2094 rad)
        - Cart position |x| > 2.4
        - Episode length > max_steps (default 200)

    Reward: +1 for each timestep the pole remains upright.

    Physics parameters:
        - Cart mass: 1.0 kg
        - Pole mass: 0.1 kg
        - Pole half-length: 0.5 m
        - Gravity: 9.8 m/s²
        - Integration timestep: 0.02 s

    Parameters
    ----------
    max_steps : int
        Maximum steps per episode.
    seed : int, optional
        Random seed for initial state perturbation.

    Examples
    --------
    >>> env = CartPoleEnv(seed=42)
    >>> state = env.reset()
    >>> state, reward, done, info = env.step(1)
    """

    def __init__(self, max_steps: int = 200, seed: Optional[int] = None):
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        # Physics constants
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.pole_half_length = 0.5
        self.force_mag = 10.0
        self.dt = 0.02  # Euler integration timestep

        # Termination thresholds
        self.theta_threshold = 12 * np.pi / 180  # 12 degrees
        self.x_threshold = 2.4

        # Env metadata
        self.n_actions = 2
        self.obs_dim = 4
        self.name = "CartPole-v0"

        # State
        self.state: Optional[np.ndarray] = None
        self.steps: int = 0

    def reset(self) -> np.ndarray:
        """
        Reset the environment to a random initial state.

        The initial state is drawn from U(-0.05, 0.05) for all components.

        Returns
        -------
        np.ndarray
            Initial observation [x, x_dot, theta, theta_dot].
        """
        self.state = self._rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take one step in the environment.

        Uses semi-implicit Euler integration of the cart-pole equations
        of motion.

        Parameters
        ----------
        action : int
            0 (push left) or 1 (push right).

        Returns
        -------
        state : np.ndarray
            New observation.
        reward : float
            Reward (+1.0 if not done).
        done : bool
            Whether the episode has ended.
        info : dict
            Additional information.
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step()")

        x, x_dot, theta, theta_dot = self.state

        # Applied force
        force = self.force_mag if action == 1 else -self.force_mag

        # Physics: equations of motion for cart-pole system
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Acceleration of the pole angular velocity
        # From Lagrangian mechanics:
        #   θ̈ = (g·sin(θ) + cos(θ)·(-F - m_p·l·θ̇²·sin(θ)) / m_total)
        #        / (l · (4/3 - m_p·cos²(θ) / m_total))
        temp = (
            force + self.mass_pole * self.pole_half_length * theta_dot ** 2 * sin_theta
        ) / self.total_mass

        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.pole_half_length
            * (4.0 / 3.0 - self.mass_pole * cos_theta ** 2 / self.total_mass)
        )

        x_acc = temp - self.mass_pole * self.pole_half_length * theta_acc * cos_theta / self.total_mass

        # Euler integration
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        # Check termination
        done = bool(
            abs(x) > self.x_threshold
            or abs(theta) > self.theta_threshold
            or self.steps >= self.max_steps
        )

        reward = 1.0 if not done else 0.0

        info = {
            "steps": self.steps,
            "x": x,
            "theta_deg": np.degrees(theta),
        }

        return self.state.copy(), reward, done, info

    def render_ascii(self) -> str:
        """
        Render the current state as ASCII art.

        Returns
        -------
        str
            ASCII representation of the cart-pole.
        """
        if self.state is None:
            return "Environment not initialized. Call reset()."

        x, _, theta, _ = self.state
        width = 40
        cart_pos = int((x / self.x_threshold + 1) * width / 2)
        cart_pos = max(2, min(width - 3, cart_pos))

        # Pole direction
        if abs(theta) < 0.05:
            pole_char = "|"
        elif theta > 0:
            pole_char = "/"
        else:
            pole_char = "\\"

        lines = []
        lines.append("=" * (width + 2))

        # Pole line
        pole_line = [" "] * (width + 2)
        pole_line[cart_pos + 1] = pole_char
        lines.append("".join(pole_line))

        # Cart line
        cart_line = [" "] * (width + 2)
        cart_line[cart_pos] = "["
        cart_line[cart_pos + 1] = "█"
        cart_line[cart_pos + 2] = "]"
        lines.append("".join(cart_line))

        # Track
        lines.append("-" * (width + 2))
        lines.append(
            f" x={x:+.2f}  θ={np.degrees(theta):+.1f}°  step={self.steps}"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CartPoleEnv(max_steps={self.max_steps})"


class FrozenLakeEnv:
    """
    FrozenLake grid world environment (4×4).

    The agent navigates a frozen lake grid to reach the goal while
    avoiding holes. The ice is optionally slippery, causing stochastic
    transitions.

    Grid layout (default):
        S F F F     S = Start
        F H F H     F = Frozen (safe)
        F F F H     H = Hole (terminal, reward=0)
        H F F G     G = Goal (terminal, reward=1)

    State: integer 0-15 (row-major position in the 4×4 grid)

    Action space: {0, 1, 2, 3}
        0 : Up
        1 : Right
        2 : Down
        3 : Left

    Slippery mode: with probability 1/3 each, the agent moves in the
    intended direction, or perpendicular left/right.

    Parameters
    ----------
    is_slippery : bool
        Whether the ice is slippery (stochastic transitions).
    max_steps : int
        Maximum steps per episode.
    custom_map : list of str, optional
        Custom 4×4 map. Characters: S, F, H, G.
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> env = FrozenLakeEnv(is_slippery=False, seed=42)
    >>> state = env.reset()
    >>> state, reward, done, info = env.step(2)  # move down
    """

    DEFAULT_MAP = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ]

    # Action names for rendering
    ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    ACTION_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    def __init__(
        self,
        is_slippery: bool = True,
        max_steps: int = 100,
        custom_map: Optional[list] = None,
        seed: Optional[int] = None,
    ):
        self.is_slippery = is_slippery
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        self.grid_map = custom_map if custom_map else self.DEFAULT_MAP
        self.nrow = len(self.grid_map)
        self.ncol = len(self.grid_map[0])

        # Env metadata
        self.n_actions = 4
        self.n_states = self.nrow * self.ncol
        self.obs_dim = self.n_states  # one-hot encoding

        self.name = "FrozenLake-v0"

        # Find start position
        self.start_pos = None
        for r, row in enumerate(self.grid_map):
            for c, cell in enumerate(row):
                if cell == "S":
                    self.start_pos = (r, c)
        if self.start_pos is None:
            self.start_pos = (0, 0)

        # State
        self.agent_pos: Optional[Tuple[int, int]] = None
        self.steps: int = 0

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert (row, col) to flat state index."""
        return pos[0] * self.ncol + pos[1]

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert flat state index to (row, col)."""
        return (state // self.ncol, state % self.ncol)

    def _get_cell(self, pos: Tuple[int, int]) -> str:
        """Get the cell type at position."""
        return self.grid_map[pos[0]][pos[1]]

    def _move(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Compute new position after taking action."""
        r, c = pos
        if action == 0:    # Up
            r = max(0, r - 1)
        elif action == 1:  # Right
            c = min(self.ncol - 1, c + 1)
        elif action == 2:  # Down
            r = min(self.nrow - 1, r + 1)
        elif action == 3:  # Left
            c = max(0, c - 1)
        return (r, c)

    def reset(self) -> np.ndarray:
        """
        Reset environment to start position.

        Returns
        -------
        np.ndarray
            One-hot encoded state vector of shape (n_states,).
        """
        self.agent_pos = self.start_pos
        self.steps = 0
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get one-hot encoded observation."""
        obs = np.zeros(self.n_states, dtype=np.float64)
        obs[self._pos_to_state(self.agent_pos)] = 1.0
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int
            Action to take (0=up, 1=right, 2=down, 3=left).

        Returns
        -------
        observation : np.ndarray
            One-hot encoded state.
        reward : float
            1.0 if goal reached, 0.0 otherwise.
        done : bool
            True if episode ended (goal, hole, or max steps).
        info : dict
            Additional information.
        """
        if self.agent_pos is None:
            raise RuntimeError("Call reset() before step()")

        # Slippery: with 1/3 probability each, move intended or ±90°
        if self.is_slippery:
            rand = self._rng.random()
            if rand < 1 / 3:
                actual_action = action
            elif rand < 2 / 3:
                actual_action = (action + 1) % 4  # Perpendicular right
            else:
                actual_action = (action - 1) % 4  # Perpendicular left
        else:
            actual_action = action

        # Move
        self.agent_pos = self._move(self.agent_pos, actual_action)
        self.steps += 1

        # Check cell type
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
        """
        Render the current grid state as ASCII art.

        Returns
        -------
        str
            ASCII grid with agent position marked.
        """
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
