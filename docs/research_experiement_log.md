# Custom Gymnasium DDA Environment

## Objective

The objective of this experiment was to implement a custom Gymnasium environment for Dynamic Difficulty Adjustment (DDA). The environment simulates a player whose probability of success decreases as game difficulty increases.

The future reinforcement learning agent will learn to dynamically adjust game difficulty to maintain a target player win rate of approximately 55%.

---

## Environment Design

### Action Space

The action represents the difficulty selected by the RL agent.

* Type: Continuous
* Range: [0, 1]
* 0 = easiest difficulty
* 1 = highest difficulty

Gymnasium representation:

`spaces.Box(low=0.0, high=1.0, shape=(1,))`

---

## Observation Space

The environment provides the RL agent with five features:

1. Current difficulty
2. Last player outcome
3. Recent win rate over the previous 10 rounds
4. Player win/loss streak
5. Episode progress (time pressure)

Observation vector:

`[difficulty, last_outcome, recent_win_rate, streak, time_pressure]`

---

## Simulated Player Model

The player's probability of success is modeled as:

`P(success) = 0.8 - 0.6 × difficulty`

Gaussian noise is added to simulate variability in player performance.

The resulting probability is clipped between 0.05 and 0.95.

This creates an inverse relationship between difficulty and player success probability.

---

## Difficulty Smoothing

Difficulty changes are smoothed using:

`new_difficulty = 0.8 × previous_difficulty + 0.2 × agent_action`

This prevents abrupt changes in game difficulty and represents a player-experience constraint.

---

## Reward Function

The total reward consists of three components:

`Reward = Performance Reward + Smoothness Reward + Engagement Reward`

### Performance Reward

The agent is encouraged to maintain the player's recent win rate close to the target win rate of 0.55.

### Smoothness Reward

Abrupt difficulty changes are penalized.

### Engagement Reward

A small positive reward is given per interaction step.

---

## Initial Sanity Check

The environment successfully executed random actions and returned valid observations, rewards, and episode termination information.

Sample output:

`Observation = [difficulty, last_outcome, recent_win_rate, streak, time_pressure]`

A debugging check was added to validate:

* Difficulty remains within [0, 1]
* Player outcome is either 0 or 1
* Recent win rate remains within [0, 1]
* Time pressure remains within [0, 1]
* Streak values remain consistent with the latest outcome

---

## Implementation Issue Identified

The initial streak update mechanism could produce unintuitive transitions between winning and losing streaks.

For example:

Winning streak:

`1 → 2 → 3`

After a loss, the previous implementation could produce:

`3 → -4`

A revised streak update mechanism will reset the streak when the outcome changes.

Expected behavior:

`Win → Win → Win → Loss → Loss`

`1 → 2 → 3 → -1 → -2`

---

## Next Step

Train a PPO reinforcement learning agent using the custom DDA environment.

The PPO agent will be evaluated against baseline difficulty strategies such as:

1. Fixed difficulty
2. Rule-based DDA
3. PPO-based DDA

