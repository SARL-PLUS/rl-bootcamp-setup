---
title: TensorBoard
---

# Reading training curves

TensorBoard is how you find out whether training is working. Being able to read
these plots critically — and to resist reading too much into them — is one of the
more transferable skills the bootcamp teaches.

!!! tip "Worked examples of these curves"
    The [worked example](worked-example.md#step-4-now-train-it-properly) shows a
    real return curve and a real set of diagnostics from a converged run, with a
    reading of each. Come back here for the reference; go there to see it applied.

## Starting it

```bash
conda activate rlbootcamp
tensorboard --logdir runs/
# open http://localhost:6006
```

Point `--logdir` at the *parent* directory, not one run. TensorBoard finds every
run beneath it and overlays them, which is the whole point.

!!! tip "Inside a notebook"
    ```python
    %load_ext tensorboard
    %tensorboard --logdir runs/
    ```

---

## The panels that matter

### `rollout/ep_rew_mean` — the only headline number

Mean episode return over recent episodes. **This is the curve.** If it is not
trending in the right direction, nothing else on the dashboard rescues you.

Note that it is a *running* mean over the last 100 episodes, so it lags: a policy
that improved a moment ago shows up here a little later.

### `rollout/ep_len_mean`

Mean episode length. How to read it depends entirely on the task:

- Where episodes end in failure (a robot falling), **longer is better**.
- Where episodes end in success (reaching a goal), **shorter is better**.
- Where there is only a time limit, it is pinned to the limit and tells you
  nothing. Pendulum is this case — it sits at 200 forever.

A sudden change here is often the most informative signal on the dashboard: it
usually means the agent's *strategy* changed qualitatively.

### `time/fps`

Throughput. Watch it for a sudden drop — that usually means something started
swapping, or a render call crept into the training loop.

### `train/*` — the losses

`actor_loss`, `critic_loss`, `value_loss`, `entropy_loss`, `approx_kl`,
`explained_variance`.

!!! danger "Losses do not mean what they mean in supervised learning"
    In supervised learning, falling loss means improvement. **In RL it does
    not.** The critic is regressing onto a target that moves as the policy
    changes, so `critic_loss` can rise while the agent gets steadily better —
    and can sit at a beautiful low value while the agent learns nothing at all.

    Judge progress by return. Use losses to diagnose *how* something is failing,
    never to decide *whether* it is.

That said, a few are genuinely diagnostic:

| Metric | What an unhealthy value looks like |
|---|---|
| `train/entropy_loss` | Collapsing toward zero very early → the policy stopped exploring and has probably locked onto a mediocre strategy. |
| `train/approx_kl` (PPO) | Consistently far above `0.02` → updates are too aggressive; lower the learning rate. |
| `train/explained_variance` | Near zero or negative → the value function is not predicting returns at all, so the advantage estimates are noise. |
| `train/clip_fraction` (PPO) | Very high (>0.3) → most updates are being clipped; the step size is too large. |

### `eval/mean_reward`

Present if you used an `EvalCallback`. This is more trustworthy than
`rollout/ep_rew_mean`, because it uses a separate environment and (usually)
deterministic actions. When the two diverge, believe this one.

---

## Reading a curve honestly

Four failure modes, in rough order of how often they catch people out:

**1. Reading a trend into noise.**
RL curves are jagged. Enable smoothing (the slider, top-left) to see the trend —
but always glance at the raw curve too. Heavy smoothing can manufacture a
convincing trend from noise. If you cannot see the improvement at smoothing 0,
be suspicious of it.

**2. Concluding anything from one run.**
Two runs differing only in seed can look completely different. Before believing
a comparison, run three seeds of each and check the curves separate by more than
they overlap. This is the most common way RL results mislead — including your
own.

**3. Comparing runs with different x-axes.**
TensorBoard will happily overlay a run measured in environment steps with one
measured in gradient updates. Compare like with like: for sample efficiency use
`total_timesteps`; for wall-clock cost use the relative-time axis.

**4. Believing reward means success.**
The agent maximises the reward function you wrote. If the curve is rising but
the behaviour is wrong, the reward function is wrong — and the curve is
faithfully reporting that. This is why you also
[watch the video](worked-example.md#step-5-watch-it).

---

## Comparing runs

Name runs so that the comparison is legible six weeks later:

```text
runs/
├── sac_lr3e-4_seed0/
├── sac_lr3e-4_seed1/
├── sac_lr1e-3_seed0/
└── sac_lr1e-3_seed1/
```

TensorBoard's run filter accepts regular expressions, so `seed0` or `lr3e-4`
immediately isolates the comparison you want. Ten seconds of naming discipline
now saves an hour of squinting later.

---

## A minimal checklist

When someone asks "is your agent working?", have an answer to each of these:

- [ ] Is `ep_rew_mean` above the **baseline** you measured before training?
- [ ] Does it hold up on a **separate evaluation** environment and unseen seeds?
- [ ] Does it hold across **several seeds**?
- [ ] Does the **behaviour** look like solving the task, not gaming the reward?

Four yeses is a result. Anything less is a hypothesis.
