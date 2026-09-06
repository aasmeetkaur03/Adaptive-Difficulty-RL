import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AdaptiveDifficultyEnv(gym.Env):
    """
    Simulated player-game loop with adjustable difficulty.

    State:
    [difficulty, last_outcome, recent_win_rate, streak, time_pressure]

    Action:
    Continuous difficulty adjustment in [0, 1]

    Reward:
    Encourages target win rate, smoothness, and engagement.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, target_win_rate=0.55, max_steps=300):
        super().__init__()

        self.target_win_rate = target_win_rate
        self.max_steps = max_steps

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32
        )

        self.step_count = 0
        self.difficulty = 0.5
        self.history = []
        self.streak = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.difficulty = 0.5
        self.history = []
        self.streak = 0

        obs = self._make_obs()

        return obs, {}

    def _simulate_player_outcome(self, difficulty):
        """
        Simulate player performance.

        Higher difficulty means lower probability of success.
        """

        base_p = 0.8 - 0.6 * difficulty

        p_success = np.clip(
            base_p + self.np_random.normal(0, 0.1),
            0.05,
            0.95
        )

        return (
            1
            if self.np_random.random() < p_success
            else 0
        )

    def _make_obs(self):
        """
        Create the observation given to the RL agent.
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

        action = np.clip(
            action,
            0.0,
            1.0
        )

        previous_difficulty = self.difficulty

        self.difficulty = float(
            0.8 * self.difficulty +
            0.2 * action[0]
        )

        outcome = self._simulate_player_outcome(
            self.difficulty
        )

        self.history.append(outcome)

        # Update streak
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

        # Recent win rate
        recent_win = float(
            np.mean(self.history[-10:])
        )

        # Reward for staying close to target
        reward_win = -abs(
            recent_win -
            self.target_win_rate
        )

        # Actual difficulty change
        difficulty_change = abs(
            self.difficulty -
            previous_difficulty
        )

        reward_smooth = (
            -0.1 *
            difficulty_change
        )

        reward_engage = 0.01

        reward = float(
            reward_win +
            reward_smooth +
            reward_engage
        )

        self.step_count += 1

        terminated = (
            self.step_count >=
            self.max_steps
        )

        truncated = False

        obs = self._make_obs()

        # Information needed for Day 4 evaluation
        info = {
            "win": int(outcome),
            "difficulty": float(self.difficulty),
            "recent_win_rate": float(recent_win),
            "difficulty_change": float(
                difficulty_change
            )
        }

        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )


if __name__ == "__main__":

    env = AdaptiveDifficultyEnv()

    obs, _ = env.reset()

    print(
        "Obs space:",
        env.observation_space
    )

    print(
        "Act space:",
        env.action_space
    )

    for _ in range(10):

        action = env.action_space.sample()

        obs, rew, term, trunc, info = env.step(
            action
        )

        if term or trunc:
            break

    print(
        "Env OK, sample obs:",
        obs
    )

    print(
        "Final Reward:",
        rew
    )
