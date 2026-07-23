#!/usr/bin/env python
"""Turn a run's TensorBoard logs into figures for the handbook.

    python scripts/plot_training.py runs/sac_Pendulum-v1_<stamp>/

Reads the TensorBoard event files directly -- the same scalars the TensorBoard
UI draws, including its exponential-moving-average smoothing -- and writes PNGs
suitable for embedding in the docs.

Two variants of every figure are written, `-light` and `-dark`, because the
handbook follows the reader's colour scheme and a single PNG cannot. MkDocs
Material picks between them with the `#only-light` / `#only-dark` URL suffixes.

Colours are not a matter of taste here: the series blue is the data-viz
palette's first categorical slot, checked against the real page surfaces for
the OKLCH lightness band, the chroma floor and >= 3:1 WCAG contrast. The site's
own accent (#58a6ff) is deliberately NOT used for data -- it fails the dark
lightness band and reaches only 2.53:1 on white.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- Theme tokens ---------------------------------------------------------
# surface / ink / series, per mode. Validated; see the module docstring.
THEMES = {
    "light": {
        "surface": "#ffffff",
        "ink": "#0b0b0b",
        "ink_muted": "#52514e",
        "grid": "#e3e3e0",
        "series": "#2a78d6",   # palette slot 1, light   4.42:1
        "rule": "#8b8b88",     # baseline reference      3.42:1
    },
    "dark": {
        "surface": "#0d1117",
        "ink": "#ffffff",
        "ink_muted": "#c3c2b7",
        "grid": "#272c33",
        "series": "#3987e5",   # palette slot 1, dark    5.20:1
        "rule": "#8b949e",     # baseline reference      6.15:1
    },
}

SOLVED_THRESHOLD = -200.0   # conventional "solved" band for Pendulum-v1


def thousands(value, _pos) -> str:
    """Axis tick formatter: 100000 -> '100k'. Keeps step labels from colliding."""
    if value == 0:
        return "0"
    return f"{value / 1000:g}k"


def load_scalars(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {tag: (steps, values)} for every scalar in the run."""
    event_files = sorted(run_dir.rglob("events.out.tfevents*"))
    if not event_files:
        raise SystemExit(f"No TensorBoard event files under {run_dir}")

    # size_guidance=0 for scalars means "load them all", not "load none".
    acc = EventAccumulator(str(event_files[-1].parent), size_guidance={"scalars": 0})
    acc.Reload()

    out = {}
    for tag in acc.Tags()["scalars"]:
        events = acc.Scalars(tag)
        out[tag] = (
            np.array([e.step for e in events]),
            np.array([e.value for e in events]),
        )
    return out


def style_axes(ax, theme: dict) -> None:
    """Recessive grid and axes; ink tokens for all text."""
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=theme["ink_muted"], labelsize=9, length=0)
    ax.xaxis.label.set_color(theme["ink_muted"])
    ax.yaxis.label.set_color(theme["ink_muted"])
    # Few, wide-spaced ticks in 'k' units — the default locator crams step
    # counts together until they run into one another.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))


def plot_return(scalars, baseline: float | None, out_stem: Path) -> None:
    """The headline figure: episode return against environment steps."""
    steps, values = scalars["rollout/ep_rew_mean"]

    for mode, theme in THEMES.items():
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
        fig.patch.set_facecolor(theme["surface"])
        style_axes(ax, theme)

        # One series, plotted as logged. No EMA overlay: ep_rew_mean is ALREADY
        # a 100-episode running mean, so a second smoothing pass only adds lag
        # -- and a lagging bold line over a faint raw one reads as a second,
        # worse series that the reader cannot identify without a legend.
        ax.plot(steps, values, color=theme["series"], linewidth=2)

        # Reference-line labels sit at the LEFT edge; the curve ends at the
        # right, and its endpoint label needs that corner to itself.
        if baseline is not None:
            ax.axhline(baseline, color=theme["rule"], linewidth=1.5,
                       linestyle=(0, (5, 4)))
            ax.annotate(f"random policy  {baseline:.0f}",
                        xy=(steps[0], baseline), xytext=(6, 8),
                        textcoords="offset points", ha="left",
                        fontsize=9, color=theme["ink_muted"])

        ax.axhline(SOLVED_THRESHOLD, color=theme["rule"], linewidth=1.5,
                   linestyle=(0, (5, 4)))
        # Right-hand end, below the rule. The curve sweeps up through the space
        # under this line on the LEFT, and settles above it on the right, so the
        # bottom-right corner is the one spot it never occupies.
        ax.annotate(f"commonly called solved  {SOLVED_THRESHOLD:.0f}",
                    xy=(steps[-1], SOLVED_THRESHOLD), xytext=(-6, -15),
                    textcoords="offset points", ha="right",
                    fontsize=9, color=theme["ink_muted"])

        # Direct-label the endpoint INSIDE the axes. Offsetting to the right
        # pushes it past the figure edge, where tight_layout clips it.
        final = values[-1]
        ax.annotate(f"{final:.0f}", xy=(steps[-1], final), xytext=(-8, 10),
                    textcoords="offset points", ha="right", fontsize=11,
                    weight="bold", color=theme["ink"])

        ax.set_xlabel("environment steps")
        ax.set_ylabel("mean episode return")
        ax.set_title("SAC on Pendulum-v1 — rollout/ep_rew_mean",
                     color=theme["ink"], fontsize=12, weight="bold",
                     loc="left", pad=12)
        ax.margins(x=0.02)

        fig.tight_layout()
        path = out_stem.with_name(f"{out_stem.name}-{mode}.png")
        fig.savefig(path, facecolor=theme["surface"])
        plt.close(fig)
        print(f"wrote {path}")


def plot_diagnostics(scalars, out_stem: Path) -> None:
    """Small multiples: the train/* signals, each on its own axis.

    Separate panels rather than one chart with two y-scales -- a dual-axis plot
    invites the reader to see a relationship in what is really just two
    arbitrary scalings.
    """
    wanted = [
        ("train/critic_loss", "critic loss"),
        ("train/actor_loss", "actor loss"),
        ("train/ent_coef", "entropy coefficient"),
    ]
    panels = [(tag, label) for tag, label in wanted if tag in scalars]
    if not panels:
        print("no train/* scalars found; skipping diagnostics figure")
        return

    for mode, theme in THEMES.items():
        fig, axes = plt.subplots(1, len(panels), figsize=(11, 3.1), dpi=200)
        axes = np.atleast_1d(axes)
        fig.patch.set_facecolor(theme["surface"])

        for ax, (tag, label) in zip(axes, panels):
            steps, values = scalars[tag]
            style_axes(ax, theme)
            # Raw, unsmoothed. The jaggedness IS the lesson on this figure.
            ax.plot(steps, values, color=theme["series"], linewidth=1.5)
            ax.set_title(label, color=theme["ink"], fontsize=10,
                         weight="bold", loc="left", pad=8)
            ax.set_xlabel("environment steps")
            ax.margins(x=0.02)

        fig.suptitle("Diagnostics — none of these is a progress signal",
                     color=theme["ink_muted"], fontsize=10, x=0.005, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = out_stem.with_name(f"{out_stem.name}-{mode}.png")
        fig.savefig(path, facecolor=theme["surface"])
        plt.close(fig)
        print(f"wrote {path}")


def random_baseline(env_id: str, episodes: int = 20, seed: int = 0) -> float:
    import gymnasium as gym
    env = gym.make(env_id)
    totals = []
    for i in range(episodes):
        env.reset(seed=seed + i)
        total = 0.0
        while True:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += float(reward)
            if terminated or truncated:
                break
        totals.append(total)
    env.close()
    return float(np.mean(totals))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()

    scalars = load_scalars(args.run_dir)
    print(f"scalars found: {', '.join(sorted(scalars))}\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline = None if args.no_baseline else random_baseline(args.env_id)
    if baseline is not None:
        print(f"random baseline: {baseline:.1f}\n")

    plot_return(scalars, baseline, args.out_dir / "pendulum-return")
    plot_diagnostics(scalars, args.out_dir / "pendulum-diagnostics")


if __name__ == "__main__":
    main()
