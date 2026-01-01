#!/usr/bin/env python3
"""
Tau Speedup Extrapolation Analysis (kv_cache + no_cache)

Produces two key diagrams:
1. gen_tokens (sequence length) vs tau throughput speedup, extrapolated to 1M
   - Shows both kv_cache (inference, tau wins) and no_cache (training-like, tau slower)
2. prompt_len (embedding dimension) vs tau speedup, extrapolated to 1M
   - Shows both modes if prompt_len varies in data

Uses log-space regression for robust long-range extrapolation.
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def log_model(x, a, b):
    """y = a + b*log(x)"""
    return a + b * np.log(x)


def power_model(x, a, b):
    """y = a * x^b"""
    return a * np.power(x, b)


def fit_and_extrapolate(x_data, y_data, max_x, model="log"):
    """
    Fit model to (x_data, y_data) and extrapolate to max_x.
    Returns: (x_extrap, y_extrap, params, r2)
    """
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    # Filter finite and positive
    mask = np.isfinite(x_data) & np.isfinite(y_data) & (x_data > 0) & (y_data > 0)
    if mask.sum() < 2:
        return None, None, None, None

    x_fit = x_data[mask]
    y_fit = y_data[mask]

    # Choose model
    if model == "log":
        func = log_model
        p0 = [1.0, 0.1]
    elif model == "power":
        func = power_model
        p0 = [1.0, 0.1]
    else:
        raise ValueError(f"Unknown model: {model}")

    try:
        params, _ = curve_fit(func, x_fit, y_fit, p0=p0, maxfev=5000)
    except:
        return None, None, None, None

    # R²
    y_pred = func(x_fit, *params)
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Extrapolate
    x_min = x_fit.min()
    x_extrap = np.geomspace(x_min, max_x, 200)
    y_extrap = func(x_extrap, *params)

    return x_extrap, y_extrap, params, r2


def plot_gen_tokens_speedup(df: pd.DataFrame, outdir: str, max_gen: int = 1_000_000):
    """
    Plot 1: gen_tokens (x) vs tau throughput speedup (y), extrapolated to max_gen.
    Shows BOTH kv_cache and no_cache on same plot.
    """
    ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {"kv_cache": "darkblue", "no_cache": "darkorange"}
    markers = {"kv_cache": "o", "no_cache": "s"}
    extrap_colors = {"kv_cache": "crimson", "no_cache": "purple"}

    for mode in ["kv_cache", "no_cache"]:
        mode_df = df[df["mode"] == mode].copy()

        if mode_df.empty:
            continue

        # Aggregate by gen_tokens
        agg = mode_df.groupby("gen_tokens", as_index=False).agg(
            {
                "tokens_per_sec_speedup_tau_over_nano": ["mean", "std", "count"],
            }
        )
        agg.columns = ["gen_tokens", "tps_speedup", "tps_speedup_std", "n"]

        x_obs = agg["gen_tokens"].to_numpy()
        y_obs = agg["tps_speedup"].to_numpy()
        y_std = agg["tps_speedup_std"].fillna(0).to_numpy()

        # Observed points
        ax.errorbar(
            x_obs,
            y_obs,
            yerr=y_std,
            fmt=markers[mode],
            markersize=10,
            capsize=5,
            color=colors[mode],
            ecolor="gray",
            linewidth=2,
            label=f"{mode} observed",
            zorder=5,
        )

        # Fit both log and power, pick best
        x_log, y_log, params_log, r2_log = fit_and_extrapolate(
            x_obs, y_obs, max_gen, model="log"
        )
        x_pow, y_pow, params_pow, r2_pow = fit_and_extrapolate(
            x_obs, y_obs, max_gen, model="power"
        )

        if r2_log is not None and r2_pow is not None:
            if r2_log >= r2_pow:
                x_extrap, y_extrap, params, r2, model_name = (
                    x_log,
                    y_log,
                    params_log,
                    r2_log,
                    "log",
                )
            else:
                x_extrap, y_extrap, params, r2, model_name = (
                    x_pow,
                    y_pow,
                    params_pow,
                    r2_pow,
                    "power",
                )
        elif r2_log is not None:
            x_extrap, y_extrap, params, r2, model_name = (
                x_log,
                y_log,
                params_log,
                r2_log,
                "log",
            )
        elif r2_pow is not None:
            x_extrap, y_extrap, params, r2, model_name = (
                x_pow,
                y_pow,
                params_pow,
                r2_pow,
                "power",
            )
        else:
            x_extrap, y_extrap, params, r2, model_name = None, None, None, None, "none"

        # Extrapolation line
        if x_extrap is not None:
            if model_name == "log":
                eq = f"y={params[0]:.2f}+{params[1]:.3f}·ln(x)"
            elif model_name == "power":
                eq = f"y={params[0]:.2f}·x^{{{params[1]:.3f}}}"
            else:
                eq = ""

            ax.plot(
                x_extrap,
                y_extrap,
                "--",
                color=extrap_colors[mode],
                linewidth=2.5,
                alpha=0.8,
                label=f"{mode} extrap ({model_name}, R²={r2:.3f})",
                zorder=3,
            )
            ax.fill_between(
                x_extrap,
                y_extrap * 0.92,
                y_extrap * 1.08,
                color=extrap_colors[mode],
                alpha=0.08,
                zorder=1,
            )

            # Save extrapolation table
            extrap_df = pd.DataFrame(
                {"gen_tokens": x_extrap, f"speedup_{mode}_predicted": y_extrap}
            )
            obs_df = pd.DataFrame(
                {"gen_tokens": x_obs, f"speedup_{mode}_observed": y_obs}
            )
            extrap_df = extrap_df.merge(obs_df, on="gen_tokens", how="left")
            extrap_df.to_csv(
                os.path.join(outdir, f"gen_tokens_extrapolation_{mode}.csv"),
                index=False,
            )
            print(f"  → gen_tokens_extrapolation_{mode}.csv")

    # Reference line at speedup=1
    ax.axhline(
        1.0,
        color="black",
        linewidth=2,
        linestyle=":",
        alpha=0.7,
        label="Break-even (speedup=1)",
        zorder=4,
    )

    ax.set_xscale("log")
    ax.set_xlabel(
        "gen_tokens (sequence length, log scale)", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel(
        "Tau throughput speedup\n(tokens/sec tau ÷ tokens/sec nano)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_title(
        "Tau Speedup vs Sequence Length (gen_tokens)\n"
        "kv_cache (inference, tau faster) vs no_cache (training-like, tau slower)\n"
        "Extrapolated to 1M tokens",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, which="both", linestyle="--")

    # Add text box explaining
    textstr = (
        "kv_cache: decode with KV-cache (inference)\n"
        "no_cache: full recompute each step (training-like)"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.6)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, "tau_speedup_vs_gen_tokens_1M_both_modes.png"), dpi=300
    )
    plt.close()
    print(f"  → tau_speedup_vs_gen_tokens_1M_both_modes.png")


def plot_prompt_len_speedup(df: pd.DataFrame, outdir: str, max_prompt: int = 1_000_000):
    """
    Plot 2: prompt_len (embedding dimension, x) vs tau speedup (y), extrapolated to max_prompt.
    Shows BOTH kv_cache and no_cache if prompt_len varies.

    NOTE: Current dataset has prompt_len=384 constant, so this will show a warning.
    """
    ensure_dir(outdir)

    # Check if prompt_len varies
    prompt_vals = df["prompt_len"].unique()
    if len(prompt_vals) < 2:
        # Not enough variation
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(
            0.5,
            0.5,
            f"⚠ Insufficient data: prompt_len is constant ({prompt_vals[0]})\n\n"
            "To generate this diagram, run experiments with varying prompt_len\n"
            "(e.g., 128, 256, 512, 1024, 2048, 4096, 8192)\n\n"
            "This dimension represents the embedding/model dimension,\n"
            "which affects prefill cost and memory complexity.",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=14,
            color="darkred",
            weight="bold",
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round", facecolor="lightyellow", edgecolor="red", linewidth=2
            ),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(
            "Tau Speedup vs Embedding Dimension (prompt_len)\n"
            "kv_cache vs no_cache • Extrapolated to 1M dimensions",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, "tau_speedup_vs_prompt_len_1M_both_modes.png"), dpi=300
        )
        plt.close()
        print(
            f"  → tau_speedup_vs_prompt_len_1M_both_modes.png (placeholder - insufficient data)"
        )
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {"kv_cache": "darkgreen", "no_cache": "brown"}
    markers = {"kv_cache": "s", "no_cache": "^"}
    extrap_colors = {"kv_cache": "purple", "no_cache": "olive"}

    for mode in ["kv_cache", "no_cache"]:
        mode_df = df[df["mode"] == mode].copy()

        if mode_df.empty:
            continue

        # Aggregate by prompt_len
        agg = mode_df.groupby("prompt_len", as_index=False).agg(
            {
                "tokens_per_sec_speedup_tau_over_nano": ["mean", "std", "count"],
            }
        )
        agg.columns = ["prompt_len", "tps_speedup", "tps_speedup_std", "n"]

        x_obs = agg["prompt_len"].to_numpy()
        y_obs = agg["tps_speedup"].to_numpy()
        y_std = agg["tps_speedup_std"].fillna(0).to_numpy()

        ax.errorbar(
            x_obs,
            y_obs,
            yerr=y_std,
            fmt=markers[mode],
            markersize=10,
            capsize=5,
            color=colors[mode],
            ecolor="gray",
            linewidth=2,
            label=f"{mode} observed",
            zorder=5,
        )

        # Fit
        x_log, y_log, params_log, r2_log = fit_and_extrapolate(
            x_obs, y_obs, max_prompt, model="log"
        )
        x_pow, y_pow, params_pow, r2_pow = fit_and_extrapolate(
            x_obs, y_obs, max_prompt, model="power"
        )

        if r2_log is not None and r2_pow is not None:
            if r2_log >= r2_pow:
                x_extrap, y_extrap, params, r2, model_name = (
                    x_log,
                    y_log,
                    params_log,
                    r2_log,
                    "log",
                )
            else:
                x_extrap, y_extrap, params, r2, model_name = (
                    x_pow,
                    y_pow,
                    params_pow,
                    r2_pow,
                    "power",
                )
        elif r2_log is not None:
            x_extrap, y_extrap, params, r2, model_name = (
                x_log,
                y_log,
                params_log,
                r2_log,
                "log",
            )
        elif r2_pow is not None:
            x_extrap, y_extrap, params, r2, model_name = (
                x_pow,
                y_pow,
                params_pow,
                r2_pow,
                "power",
            )
        else:
            x_extrap, y_extrap, params, r2, model_name = None, None, None, None, "none"

        if x_extrap is not None:
            if model_name == "log":
                eq = f"y={params[0]:.2f}+{params[1]:.3f}·ln(x)"
            elif model_name == "power":
                eq = f"y={params[0]:.2f}·x^{{{params[1]:.3f}}}"
            else:
                eq = ""

            ax.plot(
                x_extrap,
                y_extrap,
                "--",
                color=extrap_colors[mode],
                linewidth=2.5,
                alpha=0.8,
                label=f"{mode} extrap ({model_name}, R²={r2:.3f})",
                zorder=3,
            )
            ax.fill_between(
                x_extrap,
                y_extrap * 0.92,
                y_extrap * 1.08,
                color=extrap_colors[mode],
                alpha=0.08,
                zorder=1,
            )

            extrap_df = pd.DataFrame(
                {"prompt_len": x_extrap, f"speedup_{mode}_predicted": y_extrap}
            )
            obs_df = pd.DataFrame(
                {"prompt_len": x_obs, f"speedup_{mode}_observed": y_obs}
            )
            extrap_df = extrap_df.merge(obs_df, on="prompt_len", how="left")
            extrap_df.to_csv(
                os.path.join(outdir, f"prompt_len_extrapolation_{mode}.csv"),
                index=False,
            )
            print(f"  → prompt_len_extrapolation_{mode}.csv")

    ax.axhline(
        1.0,
        color="black",
        linewidth=2,
        linestyle=":",
        alpha=0.7,
        label="Break-even (speedup=1)",
        zorder=4,
    )

    ax.set_xscale("log")
    ax.set_xlabel(
        "prompt_len (embedding dimension, log scale)", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel(
        "Tau throughput speedup\n(tokens/sec tau ÷ tokens/sec nano)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_title(
        "Tau Speedup vs Embedding Dimension (prompt_len)\n"
        "kv_cache vs no_cache • Extrapolated to 1M dimensions",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, which="both", linestyle="--")

    textstr = (
        "kv_cache: decode with KV-cache (inference)\n"
        "no_cache: full recompute each step (training-like)"
    )
    props = dict(boxstyle="round", facecolor="lightgreen", alpha=0.6)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(outdir, "tau_speedup_vs_prompt_len_1M_both_modes.png"), dpi=300
    )
    plt.close()
    print(f"  → tau_speedup_vs_prompt_len_1M_both_modes.png")


def main():
    ap = argparse.ArgumentParser(
        description="Tau speedup extrapolation to 1M context/dimensions (kv_cache + no_cache)"
    )
    ap.add_argument(
        "--pairwise_csv",
        default="tau_vs_nano__pairwise_speedups.csv",
        help="Path to tau_vs_nano__pairwise_speedups.csv",
    )
    ap.add_argument("--outdir", default="plots_extrapolation", help="Output directory")
    ap.add_argument(
        "--max_gen_tokens",
        type=int,
        default=1_000_000,
        help="Extrapolate gen_tokens to this value",
    )
    ap.add_argument(
        "--max_prompt_len",
        type=int,
        default=1_000_000,
        help="Extrapolate prompt_len to this value",
    )
    args = ap.parse_args()

    ensure_dir(args.outdir)

    print(f"Loading {args.pairwise_csv}...")
    df = pd.read_csv(args.pairwise_csv)
    print(f"  Loaded {len(df)} pairwise speedup rows")
    print(f"  Modes: {df['mode'].unique()}")
    print(f"  gen_tokens range: {df['gen_tokens'].min()} - {df['gen_tokens'].max()}")
    print(f"  prompt_len values: {sorted(df['prompt_len'].unique())}")

    print("\nGenerating extrapolation plots (kv_cache + no_cache)...")
    plot_gen_tokens_speedup(df, args.outdir, args.max_gen_tokens)
    plot_prompt_len_speedup(df, args.outdir, args.max_prompt_len)

    print(f"\n✓ All outputs written to: {args.outdir}/")
    print("Key files:")
    print("  - tau_speedup_vs_gen_tokens_1M_both_modes.png")
    print("  - tau_speedup_vs_prompt_len_1M_both_modes.png")
    print("  - gen_tokens_extrapolation_kv_cache.csv")
    print("  - gen_tokens_extrapolation_no_cache.csv")
    print("  - prompt_len_extrapolation_*.csv (if prompt_len varies)")


if __name__ == "__main__":
    main()
