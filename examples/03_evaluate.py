#!/usr/bin/env python
"""Step 3 — evaluate a saved agent, and compare it against random.

    python examples/03_evaluate.py runs/sac_Pendulum-v1_.../model.zip
    python examples/03_evaluate.py runs/.../model.zip --video

Reporting a trained agent's score on its own is close to meaningless. This
script always prints the random-policy baseline next to it, because "is the
agent better than doing nothing sensible?" is the first question worth asking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

ALGOS = {"sac": SAC, "ppo": PPO}


class HangingStart(gym.Wrapper):
    """Force Pendulum to start hanging straight down, at rest.

    Pendulum-v1 randomises the initial angle, so the difficulty of an episode is
    luck. This pins it to the hardest case: the bottom of the potential well
    (theta = pi) with zero velocity. The motor is too weak to lift the weight
    from there, so the policy has no option but to pump energy in first.

    Applied INSIDE RecordVideo so the recorder's first captured frame already
    shows the overridden state.
    """

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        unwrapped = self.env.unwrapped
        unwrapped.state = np.array([np.pi, 0.0])   # [theta, theta_dot]
        return unwrapped._get_obs(), info


def random_baseline(env_id: str, episodes: int, seed: int) -> tuple[float, float]:
    """Mean and std return of a uniformly random policy."""
    env = gym.make(env_id)
    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total = 0.0
        while True:
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += float(reward)
            if terminated or truncated:
                break
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, help="path to a saved .zip model")
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--algo", default="sac", choices=sorted(ALGOS))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100, help="eval seed; keep it "
                        "DIFFERENT from the training seed")
    parser.add_argument("--video", action="store_true", help="record an .mp4 (needs ffmpeg)")
    parser.add_argument("--start-hanging", action="store_true",
                        help="record from the hardest start: hanging down, at rest")
    parser.add_argument("--video-name", default="rl-video",
                        help="filename prefix for the recorded episode")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise SystemExit(f"No such model: {args.model_path}")

    model = ALGOS[args.algo].load(args.model_path, device="cpu")

    # --- The trained agent -------------------------------------------------
    # deterministic=True takes the policy's mean action instead of sampling.
    # Report both: a large gap between them tells you the policy is still very
    # stochastic, which matters when you deploy it.
    env = gym.make(args.env_id)
    det_mean, det_std = evaluate_policy(
        model, env, n_eval_episodes=args.episodes, deterministic=True, warn=False
    )
    sto_mean, sto_std = evaluate_policy(
        model, env, n_eval_episodes=args.episodes, deterministic=False, warn=False
    )
    env.close()

    # --- The baseline ------------------------------------------------------
    rnd_mean, rnd_std = random_baseline(args.env_id, args.episodes, args.seed)

    print(f"\n{args.env_id} — {args.episodes} episodes\n")
    print(f"  {'policy':<22} {'mean return':>14}  {'std':>8}")
    print(f"  {'-' * 46}")
    print(f"  {'random baseline':<22} {rnd_mean:>14.1f}  {rnd_std:>8.1f}")
    print(f"  {'agent (stochastic)':<22} {sto_mean:>14.1f}  {sto_std:>8.1f}")
    print(f"  {'agent (deterministic)':<22} {det_mean:>14.1f}  {det_std:>8.1f}")

    improvement = det_mean - rnd_mean
    print(f"\n  agent - random = {improvement:+.1f}")
    if improvement <= 0:
        print("  The agent does NOT beat random. Train longer, or something is wrong.")

    # --- Optional video ----------------------------------------------------
    if args.video:
        video_dir = args.model_path.parent / "videos"
        env = gym.make(args.env_id, render_mode="rgb_array")
        if args.start_hanging:
            env = HangingStart(env)
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(video_dir), name_prefix=args.video_name,
            episode_trigger=lambda ep: ep == 0
        )
        obs, _ = env.reset(seed=args.seed)
        episode_return = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            if terminated or truncated:
                break
        env.close()
        start = "hanging down, at rest" if args.start_hanging else "random"
        print(f"\n  Video written to {video_dir}/  (start: {start}, "
              f"episode return {episode_return:.1f})")


if __name__ == "__main__":
    main()
