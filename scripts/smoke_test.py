#!/usr/bin/env python
"""Verify that the `rlbootcamp` environment is ready for the tutorials.

Run this once you have created the environment, and again on the morning of the
bootcamp:

    conda activate rlbootcamp
    python scripts/smoke_test.py

Every check runs independently, so a single failure does not hide the others.
The script exits with status 0 only if everything passed. If something fails,
copy the WHOLE output and take it to the troubleshooting page (or an organiser).
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
import traceback

# (module name, human label, why the tutorials need it)
PACKAGES = [
    ("numpy", "NumPy", "arrays and maths"),
    ("matplotlib", "Matplotlib", "plots and rendering"),
    ("torch", "PyTorch", "neural networks"),
    ("gymnasium", "Gymnasium", "the environment API"),
    ("stable_baselines3", "Stable-Baselines3", "RL algorithms"),
    ("sb3_contrib", "SB3-Contrib", "extra algorithms (masking, TRPO)"),
    ("mujoco", "MuJoCo", "physics simulation"),
    ("tensorboard", "TensorBoard", "training curves"),
]

PASS = "  ok  "
FAIL = " FAIL "
WARN = " warn "

results: list[tuple[str, str]] = []


def record(status: str, label: str, detail: str = "") -> None:
    """Print one result line and remember it for the final summary."""
    line = f"[{status}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    results.append((status, label))


def header(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


# --------------------------------------------------------------------------
# 1. Interpreter
# --------------------------------------------------------------------------
header("1. Python interpreter")

print(f"       executable : {sys.executable}")
print(f"       version    : {platform.python_version()}")
print(f"       platform   : {platform.system()} {platform.machine()}")

if sys.version_info < (3, 10):
    record(FAIL, "Python version", f"{platform.python_version()} is too old, need >= 3.10")
elif sys.version_info >= (3, 14):
    record(WARN, "Python version", f"{platform.python_version()} is newer than tested (3.12)")
else:
    record(PASS, "Python version", platform.python_version())

if "rlbootcamp" not in sys.executable:
    record(
        WARN,
        "Environment",
        "interpreter path has no 'rlbootcamp' in it — did you `conda activate rlbootcamp`?",
    )
else:
    record(PASS, "Environment", "running inside rlbootcamp")

# Apple Silicon users occasionally end up on an emulated x86_64 stack.
if platform.system() == "Darwin" and platform.machine() == "x86_64":
    record(
        WARN,
        "Apple Silicon",
        "Python reports x86_64 — if this is an M-series Mac you are running under "
        "Rosetta; see the troubleshooting page",
    )

# --------------------------------------------------------------------------
# 2. Packages
# --------------------------------------------------------------------------
header("2. Packages")

for module_name, label, purpose in PACKAGES:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "?")
        record(PASS, f"{label:<20}", f"{version}  ({purpose})")
    except Exception as exc:  # noqa: BLE001 — we want to report anything at all
        record(FAIL, f"{label:<20}", f"{type(exc).__name__}: {exc}")

# ffmpeg is a binary, not a Python package — videos silently fail without it.
if shutil.which("ffmpeg"):
    record(PASS, f"{'ffmpeg':<20}", "found on PATH  (saving .mp4 videos)")
else:
    record(FAIL, f"{'ffmpeg':<20}", "not on PATH — video export will not work")

# --------------------------------------------------------------------------
# 3. A classic-control environment steps
# --------------------------------------------------------------------------
header("3. Stepping a classic-control environment")

try:
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()
    record(PASS, "CartPole-v1", f"obs shape {obs.shape}, reward {reward}")
except Exception as exc:  # noqa: BLE001
    record(FAIL, "CartPole-v1", f"{type(exc).__name__}: {exc}")
    traceback.print_exc()

# --------------------------------------------------------------------------
# 4. MuJoCo physics
# --------------------------------------------------------------------------
header("4. MuJoCo physics")

try:
    import gymnasium as gym

    env = gym.make("Ant-v5")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()
    record(PASS, "Ant-v5", f"obs shape {obs.shape}, action shape {env.action_space.shape}")
except Exception as exc:  # noqa: BLE001
    record(FAIL, "Ant-v5", f"{type(exc).__name__}: {exc}")
    traceback.print_exc()

# --------------------------------------------------------------------------
# 5. A tiny training run really learns
# --------------------------------------------------------------------------
header("5. Training (this takes ~15-30 seconds)")

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy

    env = gym.make("CartPole-v1")
    # device="cpu" is deliberate: SB3 warns (correctly) that small MLP policies
    # are slower on a GPU than on the CPU. Every bootcamp exercise is CPU-sized.
    model = PPO("MlpPolicy", env, verbose=0, seed=0, device="cpu")

    model.learn(total_timesteps=5_000, progress_bar=False)
    mean_return, _ = evaluate_policy(model, env, n_eval_episodes=5, warn=False)
    env.close()

    # We are testing that the machinery runs, NOT that the agent is any good:
    # 5k steps is far too few to conclude anything, and CartPole is easy enough
    # that even an untrained policy sometimes scores well.
    record(PASS, "PPO on CartPole", f"trained 5k steps, mean return {mean_return:.0f}")
except Exception as exc:  # noqa: BLE001
    record(FAIL, "PPO on CartPole", f"{type(exc).__name__}: {exc}")
    traceback.print_exc()

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
failures = [label for status, label in results if status == FAIL]
warnings = [label for status, label in results if status == WARN]

header("Summary")

if failures:
    print(f"{len(failures)} check(s) FAILED:")
    for label in failures:
        print(f"  - {label.strip()}")
    print(
        "\nSee https://sarl-plus.github.io/rl-bootcamp-setup/setup/troubleshooting/"
        "\nIf that does not help, bring this entire output to an organiser."
    )
    sys.exit(1)

if warnings:
    print(f"Passed, with {len(warnings)} warning(s):")
    for label in warnings:
        print(f"  - {label.strip()}")
    print()

print("Everything works. You are ready for the bootcamp.")
sys.exit(0)
