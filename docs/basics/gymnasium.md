---
title: Gymnasium basics
---

# Gymnasium in one page

[Gymnasium](https://gymnasium.farama.org/) is the standard API for RL
environments — the successor to OpenAI Gym. Every environment you meet at the
bootcamp, including ones you write yourself, speaks this interface.

There is remarkably little to it.

---

## The loop

```python
import gymnasium as gym

env = gym.make("Pendulum-v1")

obs, info = env.reset(seed=0)          # (1)!

while True:
    action = env.action_space.sample()  # (2)!
    obs, reward, terminated, truncated, info = env.step(action)  # (3)!

    if terminated or truncated:         # (4)!
        break

env.close()
```

1. `reset` starts a new episode and returns the first observation. Passing a
   `seed` makes the episode sequence reproducible.
2. **This is the only line an RL algorithm replaces.** A policy maps `obs` to an
   action instead of sampling blindly.
3. `step` advances the simulation by one timestep.
4. Two different end conditions — see below. Getting these confused is a real bug.

That is the entire API surface you need. Everything else is convenience.

---

## `terminated` vs `truncated`

This split is the one piece of the API that trips people up, and it matters for
correctness, not just tidiness.

| Flag | Meaning | Example |
|---|---|---|
| `terminated` | The episode ended **because of the task itself**: a goal was reached, the robot fell over, the game was lost. | The pole fell past the angle limit. |
| `truncated` | The episode was **cut short externally**, usually by a time limit. The task had not actually finished. | 200 steps elapsed. |

Why it matters: value-based methods bootstrap the value of the next state. When
an episode `terminated`, there genuinely is no future, so the target is just the
reward. When it was merely `truncated`, the future still exists and must be
bootstrapped. Treating truncation as termination systematically biases value
estimates downward — a silent bug that shows up only as "my agent is worse than
it should be".

Stable-Baselines3 handles this correctly for you. Your own code will not, unless
you write it that way.

---

## Spaces

Every environment declares what its observations and actions look like:

```python
env.observation_space    # Box(-8.0, 8.0, (3,), float32)
env.action_space         # Box(-2.0, 2.0, (1,), float32)
```

The two you will use constantly:

- **`Box`** — continuous, with bounds and a shape. Joint torques, positions,
  velocities, sensor readings.
- **`Discrete(n)`** — an integer in `0..n-1`. "Move left / move right", or a menu
  of commands.

Also common: `MultiDiscrete` (several independent discrete choices — e.g. one
command per agent), `MultiBinary`, and `Dict` (named sub-spaces, useful when an
observation has genuinely different parts).

```python
env.action_space.sample()       # a valid random action
env.action_space.contains(a)    # is `a` legal?
env.observation_space.shape     # (3,)
```

!!! tip "Check the space before you debug the algorithm"
    A large fraction of "my agent won't learn" turns out to be a mismatch between
    what the environment provides and what the policy expects — wrong shape,
    unnormalised scale, or a `Box` that should have been `Discrete`. Print both
    spaces first. It costs one line.

---

## Wrappers

A wrapper is an environment that wraps another environment and changes one thing.
They compose, and they are the idiomatic way to modify behaviour without touching
the original code:

```python
from gymnasium.wrappers import TimeLimit, RecordVideo, NormalizeObservation

env = gym.make("Pendulum-v1", render_mode="rgb_array")
env = TimeLimit(env, max_episode_steps=200)      # truncate after 200 steps
env = NormalizeObservation(env)                   # running mean/std normalisation
env = RecordVideo(env, video_folder="videos")     # save episodes as .mp4
```

Useful ones to know: `TimeLimit`, `RecordVideo`, `RecordEpisodeStatistics`,
`NormalizeObservation`, `NormalizeReward`, `ClipAction`, `FrameStackObservation`.

You can reach the underlying environment at any time with `env.unwrapped`.

!!! warning "Normalisation statistics are part of your model"
    `NormalizeObservation` learns a running mean and standard deviation while
    training. If you save the policy but not those statistics, the policy will
    receive differently-scaled inputs at evaluation time and appear to have
    forgotten everything. Save and reload them together.

---

## Render modes

Set once, at construction:

```python
gym.make("Pendulum-v1")                          # no rendering — fastest
gym.make("Pendulum-v1", render_mode="human")     # live window (needs a display)
gym.make("Pendulum-v1", render_mode="rgb_array") # frames as arrays, for video
```

Use `None` for training and `rgb_array` for recording. `human` is for
interactive debugging and does not work over plain SSH.

---

## Writing your own environment

Any class with this shape is a Gymnasium environment:

```python
import gymnasium as gym
import numpy as np


class MyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)          # seeds self.np_random — always call this
        self.state = self.np_random.uniform(-1, 1, size=2).astype(np.float32)
        return self.state, {}             # (observation, info)

    def step(self, action):
        # ... advance the simulation ...
        reward = float(-np.linalg.norm(self.state))
        terminated = bool(np.linalg.norm(self.state) < 0.05)
        truncated = False
        return self.state, reward, terminated, truncated, {}

    def render(self):
        ...

    def close(self):
        ...
```

Two habits worth forming immediately:

- **Use `self.np_random`**, seeded by `super().reset(seed=seed)`, rather than
  `np.random` directly. Otherwise your environment ignores seeds and your results
  are not reproducible.
- **Check your own environment** with the built-in validator, which catches a
  long list of subtle API violations:

  ```python
  from gymnasium.utils.env_checker import check_env
  check_env(MyEnv())
  ```

Designing an environment — deciding what goes into the observation, what the
agent is allowed to do, and how reward is shaped — is a genuine engineering
skill, and one the bootcamp spends real time on.

---

Next: [Stable-Baselines3 →](stable-baselines3.md)
