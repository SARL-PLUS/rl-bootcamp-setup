#!/usr/bin/env python
"""Step 2 — train an agent with Stable-Baselines3.

    python examples/02_train.py                    # default: SAC, 20k steps
    python examples/02_train.py --timesteps 50000
    python examples/02_train.py --algo ppo

Writes a TensorBoard log and a saved model into runs/<algo>_<timestamp>/.
Watch it learn with:

    tensorboard --logdir runs/

Pendulum-v1 is a continuous-control task: swing an under-powered pendulum
upright and hold it there. Rewards are always negative (a cost), so "better"
means "closer to zero".
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

ALGOS = {"sac": SAC, "ppo": PPO}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--algo", default="sac", choices=sorted(ALGOS))
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{args.algo}_{args.env_id}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Monitor records episode returns and lengths. Without it, TensorBoard has
    # no rollout/ep_rew_mean curve — which is the curve you actually care about.
    env = Monitor(gym.make(args.env_id))

    algo_cls = ALGOS[args.algo]
    model = algo_cls(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        # Small MLP policies genuinely run faster on CPU than on GPU; the data
        # transfer costs more than the matrix multiplies save.
        device="cpu",
        tensorboard_log=str(run_dir),
    )

    print(f"\nTraining {args.algo.upper()} on {args.env_id} for {args.timesteps:,} steps")
    print(f"Run directory: {run_dir}\n")

    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    model_path = run_dir / "model.zip"
    model.save(model_path)
    env.close()

    print(f"\nSaved model to {model_path}")
    print(f"Inspect the curves with:  tensorboard --logdir runs/")
    print(f"Evaluate it with:         python examples/03_evaluate.py {model_path}")


if __name__ == "__main__":
    main()
