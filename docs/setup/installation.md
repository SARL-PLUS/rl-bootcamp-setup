---
title: Installation
---

# Installation

This page gets you from a fresh laptop to a working RL environment. **Do this
before the bootcamp** — the downloads (PyTorch, MuJoCo) are large and venue Wi-Fi
cannot serve a whole room at once.

!!! abstract "What you'll end up with"
    A Conda environment called **`rlbootcamp`** containing Python 3.12,
    Gymnasium, MuJoCo, Stable-Baselines3, PyTorch and everything the tutorials
    need — plus a one-command smoke test that proves it works.

    Budget about 30 minutes, most of it downloading.

---

## 0. Prerequisites

Three things: **git**, a **Conda** distribution, and **~5 GB** of free disk.

We recommend [**Miniforge**](https://github.com/conda-forge/miniforge) — a
minimal, `conda-forge`-first Conda. [Miniconda](https://docs.conda.io/projects/miniconda/)
and Anaconda also work.

---

## 1. Install Conda

=== ":material-linux: Linux"

    ```bash
    # Download and run the Miniforge installer
    wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3.sh -b -p "$HOME/miniforge3"

    # Make conda available in your shell
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda init bash      # or: conda init zsh
    ```

    Close and reopen your terminal afterwards.

=== ":material-apple: macOS"

    The installer auto-detects Apple Silicon (`arm64`) vs Intel (`x86_64`).

    ```bash
    # Homebrew (easiest)
    brew install miniforge
    conda init zsh        # macOS default shell is zsh

    # ...or the official installer:
    curl -L -o Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3.sh -b -p "$HOME/miniforge3"
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda init zsh
    ```

    Close and reopen Terminal afterwards.

=== ":material-microsoft-windows: Windows"

    1. Download the **Miniforge3 Windows x86_64** installer from the
       [releases page](https://github.com/conda-forge/miniforge/releases/latest).
    2. Run it and accept the defaults.
    3. From the Start menu open **"Miniforge Prompt"** — a preconfigured
       terminal. Run every later command there, not in plain `cmd`.

    !!! tip "Prefer WSL2 if you can"
        MuJoCo rendering and `ffmpeg` behave better under **WSL2 (Ubuntu)**. If
        you have it, open an Ubuntu shell and follow the **Linux** tab instead.
        See [troubleshooting](troubleshooting.md#windows-wsl2) for the one-command
        WSL2 setup.

---

## 2. Get this repository

```bash
git clone https://github.com/SARL-PLUS/rl-bootcamp-setup.git
cd rl-bootcamp-setup
```

!!! note "This is not the tutorial code"
    This repository contains the environment definition, the smoke test and the
    examples used on this site. The tutorial exercises for your edition are
    distributed separately at the event — you do not need them to get set up,
    and that is the point: you can be completely ready before they exist.

---

## 3. Create the `rlbootcamp` environment

One command builds the whole thing:

```bash
conda env create -f environment.yml
conda activate rlbootcamp
```

This takes 5–15 minutes depending on your connection. Go and make coffee.

??? note "What this installs"
    Python 3.12 · NumPy · pandas · Matplotlib · PyYAML · **ffmpeg** ·
    JupyterLab · Gymnasium (+ MuJoCo and classic-control extras) · MuJoCo ·
    Stable-Baselines3 · sb3-contrib · PyTorch · Hydra · TensorBoard · moviepy ·
    pytest. The full, commented list is in `environment.yml` at the repository
    root.

!!! warning "Always work inside the environment"
    Every command on this site assumes the environment is active. Either run
    `conda activate rlbootcamp` first, or prefix one-off commands with
    `conda run -n rlbootcamp ...`. See
    [Working in the Conda environment](conda-environment.md).

### Alternative: `pip` only, no Conda

If you must avoid Conda, use a virtual environment. **You then have to install
`ffmpeg` yourself** (see [troubleshooting](troubleshooting.md#ffmpeg)).

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install "gymnasium[mujoco,classic_control]>=1.0" mujoco \
            "stable-baselines3>=2.3" sb3-contrib torch \
            tensorboard hydra-core moviepy tqdm pytest \
            numpy pandas matplotlib pyyaml jupyterlab ipykernel
```

---

## 4. Smoke test :material-test-tube:

This is the step that matters. Run it from the repository root:

```bash
python scripts/smoke_test.py
```

It checks your interpreter, every required package, `ffmpeg`, that a
classic-control environment steps, that **MuJoCo** physics runs, and that a short
PPO training run completes. Each check is independent, so one failure does not
mask the rest.

Expected output:

```text
1. Python interpreter
---------------------
       executable : /home/you/miniforge3/envs/rlbootcamp/bin/python
       version    : 3.12.12
       platform   : Linux x86_64
[  ok  ] Python version — 3.12.12
[  ok  ] Environment — running inside rlbootcamp

2. Packages
-----------
[  ok  ] NumPy                — 2.3.4  (arrays and maths)
[  ok  ] Matplotlib           — 3.10.6  (plots and rendering)
[  ok  ] PyTorch              — 2.12.0  (neural networks)
[  ok  ] Gymnasium            — 1.2.3  (the environment API)
[  ok  ] Stable-Baselines3    — 2.8.0  (RL algorithms)
[  ok  ] SB3-Contrib          — 2.8.0  (extra algorithms (masking, TRPO))
[  ok  ] MuJoCo               — 3.9.0  (physics simulation)
[  ok  ] TensorBoard          — 2.20.0  (training curves)
[  ok  ] ffmpeg               — found on PATH  (saving .mp4 videos)

3. Stepping a classic-control environment
-----------------------------------------
[  ok  ] CartPole-v1 — obs shape (4,), reward 1.0

4. MuJoCo physics
-----------------
[  ok  ] Ant-v5 — obs shape (105,), action shape (8,)

5. Training (this takes ~15-30 seconds)
---------------------------------------
[  ok  ] PPO on CartPole — trained 5k steps, mean return 381

Summary
-------
Everything works. You are ready for the bootcamp.
```

Your version numbers will differ — that is fine. What matters is that every line
says `ok` and the last line is **`Everything works.`**

If anything says `FAIL`, go to [**Troubleshooting**](troubleshooting.md) with the
whole output.

---

## 5. Run the worked example (recommended)

Setup being correct is not the same as you being ready. Spend half an hour on the
[**worked example**](../basics/worked-example.md): it trains a real agent,
compares it against a baseline, and records a video — the same shape as every
exercise you will do at the bootcamp.

---

## 6. Keep it fresh

Environments rot. Re-run the smoke test **the night before the event**:

```bash
conda activate rlbootcamp
python scripts/smoke_test.py
```

If this repository has changed in the meantime:

```bash
git pull
conda env update -f environment.yml --prune
```

---

Next: [Working in the Conda environment →](conda-environment.md)
