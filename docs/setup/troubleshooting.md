---
title: Troubleshooting
---

# Troubleshooting

Fixes for problems people actually hit, grouped by symptom. Each entry says which
OS it applies to. If yours is not here, run the
[smoke test](installation.md#4-smoke-test) and bring the **entire** output to an
organiser.

!!! tip "First, the 90% fix"
    Most "it doesn't work" reports are one of three things:

    1. The environment isn't active → `conda activate rlbootcamp`.
    2. You're in the wrong directory → run from the **repository root**.
    3. The environment is stale → `conda env update -f environment.yml --prune`.

---

## Imports and environments

### `ModuleNotFoundError` for `torch`, `gymnasium`, `stable_baselines3`…

**All OSes.** The environment isn't active, or wasn't fully created.

```bash
conda activate rlbootcamp
conda env update -f environment.yml --prune
```

### Packages "missing" even though you definitely installed them

**All OSes.** You're using a different interpreter — system Python, the `base`
environment, or another venv. Verify:

```bash
conda run -n rlbootcamp python -c "import sys; print(sys.executable)"
```

The path must contain `envs/rlbootcamp`. If you are in a notebook, check the
kernel in the top-right corner instead — see
[the notebook kernel note](conda-environment.md#use-it-in-jupyter).

### `No module named 'sb3_contrib'`

**All OSes.** `sb3-contrib` provides extra algorithms (`MaskablePPO`, `TRPO`,
`QR-DQN`). It is in `environment.yml`; if your environment predates it:

```bash
pip install sb3-contrib
```

### `conda env create` hangs on "Solving environment"

**All OSes.** The classic solver can take a very long time. Either wait it out,
or switch to the much faster libmamba solver:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Miniforge uses a fast solver by default, which is one reason we recommend it.

---

## Rendering and video

### `DependencyNotInstalled: pygame is not installed`

**All OSes.** Classic-control environments (CartPole, Pendulum, MountainCar)
need `pygame` to render — including when recording video, not just for on-screen
windows. Training works without it, which is why this often surfaces late.

```bash
pip install "gymnasium[classic_control]"
```

### Training works but `render()` or saving `.mp4` fails {#ffmpeg}

**All OSes.** You're missing **ffmpeg**, the binary that writes video files. It
is not a Python package, so `pip list` will not show the problem.

=== ":material-linux: Linux"

    ```bash
    conda install -n rlbootcamp -c conda-forge ffmpeg
    # or system-wide: sudo apt-get install -y ffmpeg
    ```

=== ":material-apple: macOS"

    ```bash
    conda install -n rlbootcamp -c conda-forge ffmpeg   # or: brew install ffmpeg
    ```

=== ":material-microsoft-windows: Windows"

    ```bash
    conda install -n rlbootcamp -c conda-forge ffmpeg
    ```

Verify with `conda run -n rlbootcamp ffmpeg -version`.

### `GLFWError` / `Failed to create GLFW window` / a black render window

This is **OpenGL**, and the fix differs per OS:

=== ":material-linux: Linux"

    Install GL libraries, and use an off-screen renderer on headless machines
    (servers, CI, laptops without a display attached):

    ```bash
    sudo apt-get install -y libgl1-mesa-glx libglew-dev libosmesa6-dev
    export MUJOCO_GL=egl      # or: osmesa   (headless software rendering)
    ```

    Add the `export` to your `~/.bashrc` to make it stick.

=== ":material-apple: macOS"

    Use the native backend:

    ```bash
    export MUJOCO_GL=glfw
    ```

    Interactive windows must run from a real Terminal session, not over plain
    SSH. On Apple Silicon also check you are on an `arm64` Python — see
    [Apple Silicon](#apple-silicon).

=== ":material-microsoft-windows: Windows"

    Native windowed rendering is finicky. Two reliable options:

    - **Don't open a live window.** Record to `.mp4` instead — you only need
      `ffmpeg`, not a GL context.
    - Run everything under [**WSL2**](#windows-wsl2) and follow the Linux tab.

---

## MuJoCo

### `mujoco` import errors, or `Ant-v5` won't build

**All OSes.** Install the extras and the binding:

```bash
pip install "gymnasium[mujoco]" mujoco
```

Modern MuJoCo (3.x) ships as a self-contained pip wheel — there is **no** manual
binary download and **no** `mjkey.txt` licence step any more. If a tutorial tells
you to set `LD_LIBRARY_PATH` or `MUJOCO_PY_*`, it is describing the old
`mujoco-py` and is out of date. Ignore it.

---

## PyTorch

### "You are trying to run PPO on the GPU, but…"

**All OSes.** This warning is correct, and harmless. Small MLP policies are
*slower* on a GPU than on a CPU, because moving data to the device costs more
than the matrix multiplies save. Pass `device="cpu"` to silence it:

```python
model = PPO("MlpPolicy", env, device="cpu")
```

### Torch is a huge download / I want a specific CUDA build

**Linux & Windows with an NVIDIA GPU.** The default wheel is fine for every
bootcamp exercise. If you specifically want a particular CUDA build, use the
selector at <https://pytorch.org/get-started/locally/>:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**A CPU-only machine is completely sufficient.** You are not disadvantaged.

### `OMP: Error #15: Initializing libiomp5...`

**All OSes, most often macOS.** Duplicate OpenMP runtimes. Workaround:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

---

## Apple Silicon (M1–M4) {#apple-silicon}

**macOS arm64.** Make sure the whole stack is native `arm64`, not Rosetta
`x86_64`:

```bash
python -c "import platform; print(platform.machine())"   # want: arm64
```

If it prints `x86_64` you installed an Intel Conda. Reinstall **Miniforge**
(its installer picks the right architecture automatically) and recreate the
environment. This is precisely why we recommend Miniforge.

---

## Windows specifics

### Long path errors during install

**Windows.** Enable long paths once, in an **admin PowerShell**:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Reboot, then recreate the environment.

### `conda` is not recognised

**Windows.** Use the **Miniforge Prompt** from the Start menu rather than plain
`cmd`/PowerShell — or run `conda init powershell` once and reopen the shell.

### Recommended: WSL2 {#windows-wsl2}

**Windows.** The smoothest path for MuJoCo and ffmpeg. In an **admin
PowerShell**:

```powershell
wsl --install -d Ubuntu
```

Reboot, set your Ubuntu username and password, then **inside Ubuntu** follow the
[Linux installation tab](installation.md#1-install-conda) from the top.

---

## Networks and disks

### Proxies and corporate networks {#proxies-and-corporate-networks}

**All OSes.** Corporate proxies and TLS-inspecting firewalls break `conda` and
`pip` with certificate errors such as `SSLError`, `CERTIFICATE_VERIFY_FAILED` or
`ProxyError`.

```bash
# Tell conda and pip about the proxy
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

If your organisation uses its own certificate authority, point the tools at its
bundle rather than disabling verification:

```bash
conda config --set ssl_verify /path/to/corporate-ca-bundle.crt
pip config set global.cert /path/to/corporate-ca-bundle.crt
```

!!! warning "Don't just turn verification off"
    `ssl_verify: false` and `pip --trusted-host` are widely suggested online.
    They work by disabling the check that protects you from tampered packages.
    Use the CA bundle instead. If you cannot, install from a **home network**
    instead and bring the working environment with you.

### Out of disk space mid-install

**All OSes.** A half-written environment is worse than none: it produces
confusing partial-import errors. Clean up and start again:

```bash
conda clean --all          # reclaims cached packages, often several GB
conda env remove -n rlbootcamp
conda env create -f environment.yml
```

On a managed machine, check your **home directory quota** — Conda installs
everything there by default.

---

## Reproducibility

### My results changed between runs

**All OSes.** Expected, and worth internalising early. RL is stochastic: network
initialisation, action sampling, and environment resets are all random. Seeding
makes a single run reproducible:

```python
model = PPO("MlpPolicy", env, seed=0)
obs, info = env.reset(seed=0)
```

But a seeded run is **one sample**, not a result. Conclusions need several seeds.
If two seeds disagree wildly, that is information about your setup, not a bug to
suppress.

Note that exact bit-for-bit reproducibility across different machines, OSes or
library versions is not guaranteed even with identical seeds.

---

## Still stuck?

1. Re-run `python scripts/smoke_test.py` and copy the **entire** output.
2. Note your **OS**, whether you used **Conda or venv**, and the output of
   `conda run -n rlbootcamp python -c "import sys; print(sys.executable)"`.
3. Open an issue on this repository, or bring it to an organiser.

Doing this **before** the event costs you five minutes. Doing it during the first
session costs you a session.
