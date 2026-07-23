# RL Bootcamp — Participant Primer

**Install the toolkit, learn the mechanics, arrive ready.**

The permanent setup-and-basics companion to the hands-on tutorial track of the
Reinforcement Learning Bootcamp.

📖 **[Read the handbook →](https://sarl-plus.github.io/rl-bootcamp-setup/)**

---

## Quick start

```bash
git clone https://github.com/SARL-PLUS/rl-bootcamp-setup.git
cd rl-bootcamp-setup
conda env create -f environment.yml
conda activate rlbootcamp
python scripts/smoke_test.py
```

If the last command prints **`Everything works. You are ready for the bootcamp.`**
you are set. If not, see [Troubleshooting](https://sarl-plus.github.io/rl-bootcamp-setup/setup/troubleshooting/).

> **Do this before the event.** PyTorch and MuJoCo are large downloads, and venue
> Wi-Fi cannot serve a full room at once.

## What's here

| | |
|---|---|
| `environment.yml` | The `rlbootcamp` Conda environment — the whole toolkit in one file. |
| `scripts/smoke_test.py` | One command that proves your machine is ready. |
| `scripts/plot_training.py` | Turns a run's TensorBoard logs into the figures used in the docs. |
| `examples/` | A complete worked example: baseline → train → evaluate → video. |
| `docs/` | The handbook source (MkDocs Material). |

### The worked example

```bash
python examples/01_random_agent.py        # measure the baseline first
python examples/02_train.py               # train SAC on Pendulum-v1 (~5-10 min)
python examples/03_evaluate.py runs/<run>/model.zip --video
```

## What's *not* here

The tutorial exercises, environments and solutions. Those are distributed to
participants at each event, in their own repository — deliberately, since the
exercises are worth more if you have not read the answers first.

This repository is intentionally **year-agnostic**: dates, venue and schedule
live on the event website for the edition you are attending.

## Building the docs locally

```bash
pip install -r requirements-docs.txt
mkdocs serve            # http://127.0.0.1:8000
```

## Contributing

Corrections and improvements are very welcome — especially troubleshooting
entries for setups we have not seen. Please keep contributions **generic**: no
edition-specific content, and nothing that reveals tutorial exercises.

## Licence

[MIT](LICENSE). If you use this to run your own workshop, attribution is
appreciated.
