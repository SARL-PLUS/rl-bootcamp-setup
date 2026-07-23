---
title: Home
---

# RL Bootcamp — Participant Primer

This is the permanent setup-and-basics companion to the hands-on tutorial track
of the **Reinforcement Learning Bootcamp**. It is deliberately small: install the
toolkit, learn the mechanics, arrive ready.

!!! info "What this site is — and is not"
    **Is:** installation for Linux/macOS/Windows, the Conda workflow, a
    troubleshooting catalogue, the Gymnasium and Stable-Baselines3 basics, and a
    complete worked example you can run today.

    **Is not:** the tutorial material. The exercises, environments and solutions
    for a given edition are released to participants at the event. Nothing on
    this site spoils them.

---

## Start here

<div class="grid cards" markdown>

-   :material-calendar-check:{ .lg .middle } **Before you arrive**

    ---

    The advice that actually matters: when to install, what hardware you need,
    what to brush up on, and the failure modes that cost people the first hour.

    [:octicons-arrow-right-24: Before you arrive](before-you-arrive.md)

-   :material-download:{ .lg .middle } **Install everything**

    ---

    OS-by-OS instructions and the one-command Conda environment, ending in a
    smoke test that proves your machine is ready.

    [:octicons-arrow-right-24: Installation](setup/installation.md)

-   :material-school:{ .lg .middle } **Learn the mechanics**

    ---

    The environment loop, the training API, and a worked example that trains a
    real agent end to end — so the live sessions are about ideas, not syntax.

    [:octicons-arrow-right-24: Basics](basics/gymnasium.md)

-   :material-bug-check:{ .lg .middle } **Something broke?**

    ---

    The fixes for the problems people actually hit: MuJoCo rendering, `ffmpeg`,
    Apple Silicon, Windows long paths, proxies and more.

    [:octicons-arrow-right-24: Troubleshooting](setup/troubleshooting.md)

</div>

---

## The 10-minute version

If you do nothing else, do this:

```bash
git clone https://github.com/SARL-PLUS/rl-bootcamp-setup.git
cd rl-bootcamp-setup
conda env create -f environment.yml
conda activate rlbootcamp
python scripts/smoke_test.py
```

If the last command prints **`Everything works. You are ready for the bootcamp.`**
you are done. If it does not, you have found your problem weeks early instead of
during the first session — go to [troubleshooting](setup/troubleshooting.md).

---

## What the tutorial track covers

Editions differ, but the skills and the toolkit are stable. Across the sessions
you can expect to work with:

| Theme | What you practise | Tools |
|---|---|---|
| **Tabular RL** | Value iteration, Q-learning, SARSA on small discrete problems | NumPy |
| **Deep RL for control** | Training continuous-control policies and reading their curves critically | Gymnasium, MuJoCo, Stable-Baselines3 |
| **Robustness** | What happens when the deployment world differs from the training world | SB3, domain randomisation |
| **Environment design** | Building an MDP yourself: observations, actions, reward, termination | Gymnasium API |
| **Quality assurance** | Baselines, seeds, ablations — proving an agent has actually learned | TensorBoard, evaluation protocol |

A theme running through the whole track: **reward going up is not evidence of
success.** You will be asked repeatedly to compare against a baseline and to
justify that RL is the right tool at all.

---

## Which parts apply to me?

- **Everyone:** [Before you arrive](before-you-arrive.md) and
  [Installation](setup/installation.md).
- **New to Gymnasium or Stable-Baselines3:** the whole [Basics](basics/gymnasium.md)
  section, and run the [worked example](basics/worked-example.md) at least once.
- **Comfortable with both:** just run the smoke test and skim
  [Before you arrive](before-you-arrive.md).

!!! tip "Edition-specific details"
    Dates, venue, schedule and registration live on the **event website** for the
    edition you are attending, not here. This site is intentionally year-agnostic
    so it stays correct.

---

## Credits

Written by **Leander Grech** with **Claude Opus 4.8**.

Last updated **23 July 2026** ·
[MIT licensed](https://github.com/SARL-PLUS/rl-bootcamp-setup/blob/main/LICENSE)
· the version is shown in the header, top right

Every command, figure and number on this site was produced by running it — the
training curves and the video come from a real 100,000-step run, not an
illustration. Corrections and additions are welcome: open an issue or a pull
request on [the repository](https://github.com/SARL-PLUS/rl-bootcamp-setup).
