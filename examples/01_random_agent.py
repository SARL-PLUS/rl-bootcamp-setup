#!/usr/bin/env python
"""Step 1 — the Gymnasium loop, with no learning at all.

Before any algorithm gets involved, make sure you can read this loop. Every RL
library you will meet is a fancier version of it.

    python examples/01_random_agent.py

Establishing what a *random* policy scores is not busywork: it is the floor that
any trained agent must beat. An agent that cannot beat random has not learned.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

EPISODES = 20
SEED = 0

# render_mode=None keeps this fast and headless. Use "human" to watch a window
# (needs a display), or "rgb_array" to capture frames — see 03_evaluate.py.
env = gym.make("Pendulum-v1")

returns: list[float] = []

for episode in range(EPISODES):
    # Seeding per episode makes this run reproducible. Do this deliberately:
    # silent nondeterminism is the single most common source of "my results
    # changed and I don't know why".
    obs, info = env.reset(seed=SEED + episode)

    episode_return = 0.0
    steps = 0

    while True:
        # THE POLICY. This is the only line an RL algorithm replaces: instead of
        # sampling blindly, it maps obs -> action.
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        episode_return += float(reward)
        steps += 1

        # terminated = the task genuinely ended (goal reached, robot fell over)
        # truncated  = we hit a time limit; the task did NOT end on its own
        # Conflating these two silently corrupts value estimates.
        if terminated or truncated:
            break

    returns.append(episode_return)
    print(f"episode {episode:2d}  steps {steps:4d}  return {episode_return:8.1f}")

env.close()

print(f"\nRandom policy over {EPISODES} episodes:")
print(f"  mean return : {np.mean(returns):8.1f}")
print(f"  std         : {np.std(returns):8.1f}")
print(f"  best / worst: {np.max(returns):8.1f} / {np.min(returns):.1f}")
print("\nThis is the number to beat. Now run 02_train.py.")
