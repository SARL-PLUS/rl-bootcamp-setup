---
title: Stable-Baselines3 basics
---

# Stable-Baselines3 in one page

[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (SB3) provides
tested implementations of the standard deep-RL algorithms. It exists so you can
spend your time on the *problem* rather than on re-deriving PPO — and so that
when something fails, you know it is your environment or your reward, not a bug
in your Adam update.

---

## Train, save, load, evaluate

The whole workflow:

```python
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("Pendulum-v1")

model = SAC("MlpPolicy", env, verbose=1, seed=0, device="cpu")
model.learn(total_timesteps=20_000)
model.save("my_agent")                    # writes my_agent.zip

model = SAC.load("my_agent", device="cpu")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"{mean_reward:.1f} +/- {std_reward:.1f}")
```

`"MlpPolicy"` is a small fully-connected network, and it is the right default for
vector observations. `"CnnPolicy"` is for images; `"MultiInputPolicy"` for `Dict`
observation spaces.

!!! note "`device="cpu"` is deliberate"
    For small MLP policies the CPU is genuinely faster — moving tensors to a GPU
    costs more than the matrix multiplies save. SB3 will warn you about this if
    you let it pick. Every bootcamp exercise is CPU-sized.

---

## Choosing an algorithm

| Algorithm | Action space | Sample efficiency | Notes |
|---|---|---|---|
| **PPO** | Discrete or continuous | Lower | On-policy. Robust, forgiving of bad hyperparameters, parallelises well. The sensible default. |
| **SAC** | Continuous only | Higher | Off-policy. Excellent on robotics-style control; the usual choice when environment steps are expensive. |
| **TD3** | Continuous only | Higher | Off-policy. Similar niche to SAC, often needs more tuning. |
| **DQN** | Discrete only | Medium | The classic. Superseded by PPO for most tasks. |
| **A2C** | Discrete or continuous | Low | Simple, mostly of pedagogical interest. |

Rules of thumb worth remembering:

- **Continuous actions and a slow simulator** → SAC.
- **Anything else, or you're not sure** → PPO.
- **Discrete actions with illegal moves** → PPO plus action masking
  (`MaskablePPO` from `sb3-contrib`).

The choice matters far less than the reward function and the observation space.
Practitioners routinely spend a day tuning an algorithm when an hour on the
reward would have been worth ten times as much.

---

## Vectorised environments

Running several copies in parallel is the cheapest speedup available for
on-policy algorithms such as PPO:

```python
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("CartPole-v1", n_envs=8)
model = PPO("MlpPolicy", vec_env, device="cpu")
model.learn(total_timesteps=100_000)
```

`total_timesteps` counts steps **across all environments**, so this is not 8×
more work — it is the same work with better-decorrelated batches and less Python
overhead.

Off-policy algorithms (SAC, TD3) benefit much less; they do a gradient update per
environment step, so the bottleneck is elsewhere.

---

## Monitoring

Wrap with `Monitor` to record episode returns and lengths. Without it,
TensorBoard has no `rollout/ep_rew_mean` curve — which is the curve you actually
care about:

```python
from stable_baselines3.common.monitor import Monitor

env = Monitor(gym.make("Pendulum-v1"))
model = SAC("MlpPolicy", env, tensorboard_log="runs/")
```

`make_vec_env` applies `Monitor` for you.

See [TensorBoard](tensorboard.md) for what the resulting curves mean.

---

## Callbacks

Callbacks hook into the training loop. The two you will use most:

```python
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

eval_callback = EvalCallback(
    Monitor(gym.make("Pendulum-v1")),   # a SEPARATE env, not the training one
    best_model_save_path="runs/best/",
    eval_freq=5_000,
    n_eval_episodes=10,
    deterministic=True,
)

checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path="runs/ckpt/")

model.learn(total_timesteps=100_000, callback=[eval_callback, checkpoint_callback])
```

!!! warning "Evaluate on a separate environment"
    Reusing the training environment for evaluation corrupts both: the
    evaluation episodes disturb the training env's internal state, and
    off-policy algorithms may treat the evaluation transitions as training data.
    Always construct a fresh one.

---

## Deterministic vs stochastic actions

```python
action, _ = model.predict(obs, deterministic=True)   # the policy's mean action
action, _ = model.predict(obs, deterministic=False)  # sampled from the policy
```

Train stochastically — exploration requires it. Evaluate both ways and report
both. A large gap between them tells you the policy is still highly stochastic,
which matters if you intend to deploy it.

---

## Reproducibility and honest reporting

```python
model = PPO("MlpPolicy", env, seed=0)
```

A seed makes one run repeatable on one machine. It does **not** make one run a
result.

!!! quote "The habit worth building now"
    Run **at least three seeds** before believing anything, and report the spread
    rather than the best. A single lucky run is the most common way RL results
    mislead — including when it is your own run misleading you.

Two further habits that carry through the whole bootcamp:

- **Always compare against a baseline.** Random, do-nothing, or a hand-written
  heuristic. A number with nothing to compare it to means nothing.
- **Evaluate on seeds you did not train on.** Otherwise you are measuring
  memorisation.

---

Next: [The worked example →](worked-example.md)
