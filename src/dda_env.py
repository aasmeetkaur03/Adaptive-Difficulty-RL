import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AdaptiveDifficultyEnv(gym.Env):
"""
Custom Gymnasium environment for Dynamic Difficulty Adjustment (DDA).

```
The environment simulates a player whose probability of success decreases
as game difficulty increases. A reinforcement learning agent continuously
adjusts the difficulty to maintain a target player win rate while avoiding
abrupt difficulty changes.

Observation:
    [difficulty,
     last_outcome,
     recent_win_rate,
     streak,
     time_pressure]

Action:
    Continuous difficulty value in the range [0, 1].

Reward:
    Combines:
    - proximity to the target win rate,
    - smoothness of difficulty changes,
    - a small engagement incentive.
"""

metadata = {"render_modes": ["human"]}

def __init__(self, target_win_rate=0.55, max_steps=300):
    """
    Initialize the Adaptive Difficulty Adjustment environment.

    Args:
        target_win_rate (float): Desired player success rate.
        max_steps (int): Maximum number of steps per episode.
    """
    super().__init__()

    self.target_win_rate = target_win_rate
    self.max_steps = max_steps

    # Action:
    # A continuous difficulty value between 0 (easiest)
    # and 1 (hardest).
    self.action_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(1,),
        dtype=np.float32
    )

    # Observation:
    # [difficulty, last_outcome, recent_win_rate, streak, time_pressure]
    self.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(5,),
        dtype=np.float32
    )

    # Environment state variables
    self.step_count = 0
    self.difficulty = 0.5
    self.history = []
    self.streak = 0

def reset(self, seed=None, options=None):
    """
    Reset the environment to its initial state.

    Returns:
        observation (np.ndarray): Initial environment observation.
        info (dict): Additional information.
    """
    super().reset(seed=seed)

    self.step_count = 0
    self.difficulty = 0.5
    self.history = []
    self.streak = 0

    observation = self._make_obs()

    return observation, {}

def _simulate_player_outcome(self, difficulty):
    """
    Simulate a player's win or loss.

    The probability of success decreases as difficulty increases.
    Random Gaussian noise is added to simulate natural variability
    in player performance.

    Args:
        difficulty (float): Current game difficulty.

    Returns:
        int: 1 for a successful outcome, 0 for failure.
    """

    # At difficulty = 0 -> base success probability = 0.8
    # At difficulty = 1 -> base success probability = 0.2
    base_success_probability = 0.8 - 0.6 * difficulty

    # Add variability to simulate inconsistent player performance.
    success_probability = np.clip(
        base_success_probability + self.np_random.normal(0, 0.1),
        0.05,
        0.95
    )

    return (
        1
        if self.np_random.random() < success_probability
        else 0
    )

def _make_obs(self):
    """
    Construct the observation provided to the RL agent.

    Returns:
        np.ndarray: Current observation vector.
    """

    recent_win_rate = (
        np.mean(self.history[-10:])
        if self.history
        else 0.5
    )

    last_outcome = (
        self.history[-1]
        if self.history
        else 0
    )

    time_pressure = min(
        self.step_count / self.max_steps,
        1.0
    )

    return np.array(
        [
            self.difficulty,
            last_outcome,
            recent_win_rate,
            self.streak,
            time_pressure
        ],
        dtype=np.float32
    )

def step(self, action):
    """
    Execute one environment step.

    The RL agent proposes a difficulty level. The environment applies
    smoothing, simulates the player's outcome, updates the player state,
    calculates the reward, and returns the next observation.

    Args:
        action (np.ndarray): Proposed difficulty value.

    Returns:
        observation (np.ndarray): Next environment observation.
        reward (float): Reward for the action.
        terminated (bool): Whether the episode has naturally ended.
        truncated (bool): Whether the episode was externally truncated.
        info (dict): Additional information.
    """

    # Ensure the action remains within the valid difficulty range.
    action = np.clip(
        action,
        self.action_space.low,
        self.action_space.high
    )

    # Store previous difficulty to measure the actual difficulty change.
    previous_difficulty = self.difficulty

    # Smooth difficulty transitions to avoid abrupt changes.
    self.difficulty = float(
        0.8 * self.difficulty +
        0.2 * action[0]
    )

    # Simulate player performance at the new difficulty.
    outcome = self._simulate_player_outcome(
        self.difficulty
    )

    # Store the latest outcome.
    self.history.append(outcome)

    # Update win/loss streak.
    # Positive values represent consecutive wins.
    # Negative values represent consecutive losses.
    if outcome == 1:

        if self.streak >= 0:
            self.streak += 1
        else:
            self.streak = 1

    else:

        if self.streak <= 0:
            self.streak -= 1
        else:
            self.streak = -1

    # Calculate recent player performance.
    recent_win_rate = np.mean(
        self.history[-10:]
    )

    # Reward component 1:
    # Encourage the player's win rate to remain close to the target.
    reward_win_rate = -abs(
        recent_win_rate - self.target_win_rate
    )

    # Reward component 2:
    # Penalize abrupt changes in actual game difficulty.
    difficulty_change = abs(
        self.difficulty - previous_difficulty
    )

    reward_smoothness = -0.1 * difficulty_change

    # Reward component 3:
    # Small incentive for sustained interaction.
    reward_engagement = 0.01

    # Total reward.
    reward = float(
        reward_win_rate +
        reward_smoothness +
        reward_engagement
    )

    # Advance the environment.
    self.step_count += 1

    # End the episode after the maximum number of steps.
    terminated = (
        self.step_count >= self.max_steps
    )

    truncated = False

    # Generate the next observation.
    observation = self._make_obs()

    return (
        observation,
        reward,
        terminated,
        truncated,
        {}
    )
```

if **name** == "**main**":

```
# Quick environment sanity check.
env = AdaptiveDifficultyEnv()

observation, _ = env.reset(seed=42)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

for step in range(10):

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(
        action
    )

    if terminated or truncated:
        break

print("\nEnvironment sanity check successful.")
print("Sample observation:", observation)
print("Final reward:", reward)
```
