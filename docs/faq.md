---
title: FAQ
---

# Frequently asked questions

## Do I need a GPU?

**No.** Every exercise is sized for a laptop CPU, and small MLP policies actually
train *faster* on CPU than on GPU — moving data to the device costs more than the
matrix multiplies save. Bring whatever laptop you have.

## Do I need to know reinforcement learning already?

No. The track starts from first principles. You do need working **Python** —
scripts, NumPy, classes, and the ability to read a traceback. See the
[self-assessment](before-you-arrive.md#self-assessment).

## Can I use Google Colab instead of installing locally?

For some parts, yes — and it is a reasonable fallback if your laptop is locked
down. But a local environment is strongly preferred: some sessions involve
rendering, longer training runs, and editing files across a project, none of
which Colab handles gracefully. If you think you will have to use Colab, tell the
organisers in advance.

## Can I use `pip`/`venv`/`uv`/Poetry instead of Conda?

Yes. Conda is the recommended path because it handles `ffmpeg` and the
cross-platform binary dependencies for you, but nothing in the tutorials requires
it. The [installation page](setup/installation.md#alternative-pip-only-no-conda)
has a plain-`pip` recipe. If you go this way, you are responsible for installing
`ffmpeg` yourself.

## Will the tutorial code be published here?

No. This repository is the permanent setup-and-basics companion. Exercise code
for each edition is distributed to participants at the event, in its own
repository. That is deliberate — the exercises are more valuable when you have
not read the solutions first.

## Which Python version?

**3.12**, as pinned in `environment.yml`. 3.10 and 3.11 work. Very new releases
often lag on wheels for PyTorch or MuJoCo, so do not reach for the newest just
because it exists.

## Why do my results change every time I run the same script?

Because RL is stochastic — network initialisation, action sampling and
environment resets are all random. Seeding makes one run repeatable:

```python
model = PPO("MlpPolicy", env, seed=0)
obs, info = env.reset(seed=0)
```

But a seeded run is one sample, not a result. See
[reproducibility](setup/troubleshooting.md#reproducibility).

## My agent's reward is going up. Is it working?

Maybe. Rising reward means the agent is getting better at maximising **the reward
function you wrote**, which is not necessarily the same as getting better at your
task. Check it against a baseline, on unseen seeds, across several runs — and
watch what it actually does. See the
[checklist](basics/tensorboard.md#a-minimal-checklist).

## How long should training take?

Longer than you expect. The [worked example](basics/worked-example.md) takes
5–10 minutes for a task you could hand-code in twenty lines. That is normal and
is not a sign that anything is wrong.

## Something failed and the troubleshooting page didn't help.

Open an issue on this repository with your OS, whether you used Conda or venv,
and the **entire** output of `python scripts/smoke_test.py`. Please do this
before the event rather than during it.

## Can I use this material to run my own workshop?

Yes — it is MIT licensed. Attribution is appreciated. If you improve something,
a pull request back would be very welcome.

## Where are the dates, venue and schedule?

On the **event website** for the edition you are attending. This site is
deliberately year-agnostic so that it stays correct across editions.
