#!/usr/bin/env python3
"""
Tau Speedup Extrapolation Analysis (kv_cache + no_cache)

Generates THREE diagrams:
  1) Speedup vs gen_tokens
  2) Speedup vs prompt_len
  3) Speedup vs embedding dimension (nembd)

Important: This script will NOT fail silently.
- If required columns are missing, it prints a clear ERROR/WARNING and:
  - Either exits (for critical missing inputs), or
  - Generates a placeholder plot (for “no variation” cases), or
  - Skips only the affected diagram but reports it loudly.

How nembd is determined:
- Default: nembd := prompt_len (works with your current pairwise speedups CSV).
  This matches your data model where prompt_len is already 384 * embd_mult.
- Optional: nembd from raw embd_mult via join (requires run_id->variant mapping):
    --nembd_source join_embd_mult --token_latencies_csv ... --runs_csv ...
  token_latencies must have (run_id, embd_mult); runs.csv must have (run_id, variant).

Input:
  --pairwise_csv tau_vs_nano__pairwise_speedups.csv

Output:
  - tau_speedup_vs_gentokens_bothmodes.png
  - tau_speedup_vs_promptlen_bothmodes.png
  - tau_speedup_vs_nembd_bothmodes.png (real or placeholder)
  - *_extrapolation_{mode}.csv
"""

import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def die(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    sys.exit(code)


def log_model(x, a, b):
    """y = a + b*ln(x)"""
    return a + b * np.log(x)


def power_model(x, a, b):
    """y = a * x^b"""
    return a * np.power(x, b)


def fit_and_extrapolate(xdata, ydata, maxx, model="log"):
    """
    Fit model to (xdata, ydata) and extrapolate to maxx.
    Returns x_extrap, y_extrap, params, r2.
    """
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)

    mask = np.isfinite(xdata) & np.isfinite(ydata) & (xdata > 0) & (ydata > 0)
    if mask.sum() < 2:
        return None, None, None, None

    xfit = xdata[mask]
    yfit = ydata[mask]

    if model == "log":
        func = log_model
        p0 = [1.0, 0.1]
    elif model == "power":
        func = power_model
        p0 = [1.0, 0.1]
    else:
        raise ValueError(f"Unknown model: {model}")

    try:
        params, _ = curve_fit(func, xfit, yfit, p0=p0, maxfev=5000)
    except Exception:
        return None, None, None, None

    ypred = func(xfit, *params)
    ssres = np.sum((yfit - ypred) ** 2)
    sstot = np.sum((yfit - yfit.mean()) ** 2)
    r2 = 1 - (ssres / sstot) if sstot > 0 else 0.0

    xmin = xfit.min()
    x_extrap = np.geomspace(xmin, maxx, 200)
    y_extrap = func(x_extrap, *params)

    return x_extrap, y_extrap, params, r2


def choose_speedup_col(df: pd.DataFrame) -> str:
    candidates = [
        "tokens_per_sec_speedup_tau_over_nano",
        "decode_total_ms_speedup_tau_over_nano",
        "total_ms_no_ttft_speedup_tau_over_nano",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    die(
        "❌ ERROR: No supported speedup column found.\n"
        f"   Looked for: {candidates}\n"
        f"   Found columns: {df.columns.tolist()}"
    )


# -----------------------------
# nembd construction
# -----------------------------
def ensure_nembd(df: pd.DataFrame, args) -> pd.DataFrame:
    """
    Ensures df has 'nembd'.
    - default: nembd := prompt_len
    - optional: join_embd_mult: requires token_latencies_csv + runs_csv with join keys.
    """
    if "nembd" in df.columns:
        return df

    if args.nembd_source == "prompt_len":
        if "prompt_len" not in df.columns:
            print(
                "\n❌ ERROR: nembd_source=prompt_len but 'prompt_len' column is missing."
            )
            return df
        out = df.copy()
        out["nembd"] = out["prompt_len"]
        print("\n✓ Derived nembd from prompt_len (nembd := prompt_len).")
        return out

    # join_embd_mult path
    needed_pairwise = {"variant"}
    if not needed_pairwise.issubset(df.columns):
        print(
            "\n❌ ERROR: nembd_source=join_embd_mult requires 'variant' in pairwise CSV."
        )
        print(f"   Found columns: {df.columns.tolist()}")
        return df

    if not args.token_latencies_csv or not args.runs_csv:
        print("\n❌ ERROR: nembd_source=join_embd_mult requires:")
        print("   --token_latencies_csv token_latencies.csv")
        print("   --runs_csv runs.csv")
        return df

    if not os.path.exists(args.token_latencies_csv):
        print(f"\n❌ ERROR: token_latencies_csv not found: {args.token_latencies_csv}")
        return df

    if not os.path.exists(args.runs_csv):
        print(f"\n❌ ERROR: runs_csv not found: {args.runs_csv}")
        return df

    tok = pd.read_csv(args.token_latencies_csv)
    runs = pd.read_csv(args.runs_csv)

    needed_tok = {"run_id", "embd_mult"}
    needed_runs = {"run_id", "variant"}

    if not needed_tok.issubset(tok.columns):
        print("\n❌ ERROR: token_latencies_csv missing required columns.")
        print(f"   Required: {sorted(needed_tok)}")
        print(f"   Found:    {tok.columns.tolist()}")
        return df

    if not needed_runs.issubset(runs.columns):
        print("\n❌ ERROR: runs_csv missing required columns.")
        print(f"   Required: {sorted(needed_runs)}")
        print(f"   Found:    {runs.columns.tolist()}")
        return df

    tok_mult = tok[["run_id", "embd_mult"]].drop_duplicates()
    tok_mult["embd_mult"] = pd.to_numeric(tok_mult["embd_mult"], errors="coerce")

    run_variant = runs[["run_id", "variant"]].drop_duplicates()

    # Map (variant -> embd_mult). If multiple embd_mult per variant exist, keep all and warn.
    vm = run_variant.merge(tok_mult, on="run_id", how="left")
    vm = vm.dropna(subset=["embd_mult"])

    # Detect conflicting embd_mult for same variant
    conflicts = vm.groupby("variant")["embd_mult"].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts) > 0:
        bad_vars = conflicts.index.tolist()
        print("\n⚠️ WARNING: Some variants map to multiple embd_mult values.")
        print(f"   Affected variants (showing up to 20): {bad_vars[:20]}")
        print("   Using the first embd_mult per variant for nembd computation.")

    vm_one = vm.sort_values(["variant", "run_id"]).drop_duplicates(
        subset=["variant"], keep="first"
    )
    out = df.merge(vm_one[["variant", "embd_mult"]], on="variant", how="left")
    out["nembd"] = args.base_nembd * out["embd_mult"]

    missing = out[out["nembd"].isna()]
    if len(missing) > 0:
        missing_vars = sorted(missing["variant"].unique().tolist())
        print(
            "\n⚠️ WARNING: Could not compute nembd for some variants (missing embd_mult join)."
        )
        print(f"   Affected variants (showing up to 20): {missing_vars[:20]}")
        print("   These rows will be excluded from the nembd plot.")

    print("\n✓ Derived nembd by joining embd_mult from raw token_latencies + runs.csv.")
    return out


# -----------------------------
# Plotting
# -----------------------------
def placeholder_plot(outpath: str, title: str, body: str):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.text(
        0.5,
        0.5,
        body,
        ha="center",
        va="center",
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
    ax.set_title(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_speedup_vs_dimension(
    df: pd.DataFrame,
    outdir: str,
    speedup_col: str,
    x_col: str,
    x_label: str,
    filename_stem: str,
    max_x: int,
):
    ensure_dir(outdir)

    if x_col not in df.columns:
        print(f"\n❌ ERROR: Cannot plot {filename_stem}: missing column '{x_col}'.")
        print(f"   Available columns: {df.columns.tolist()}")
        outpath = os.path.join(outdir, f"tau_speedup_vs_{filename_stem}_bothmodes.png")
        placeholder_plot(
            outpath,
            title=f"Tau Speedup vs {x_label}",
            body=f"Missing required column: '{x_col}'\n\nCannot generate this diagram.",
        )
        print(f"  → {os.path.basename(outpath)} (placeholder)")
        return False

    x_vals = df[x_col].dropna().unique()
    if len(x_vals) < 2:
        outpath = os.path.join(outdir, f"tau_speedup_vs_{filename_stem}_bothmodes.png")
        placeholder_plot(
            outpath,
            title=f"Tau Speedup vs {x_label}",
            body=(
                f"Insufficient data: '{x_col}' has < 2 unique values.\n"
                f"Unique values: {sorted([int(v) for v in x_vals]) if len(x_vals) else '[]'}\n\n"
                "Run experiments with varying values to generate this diagram."
            ),
        )
        print(
            f"\n⚠️ WARNING: Cannot extrapolate {filename_stem}; insufficient x variation."
        )
        print(f"  → {os.path.basename(outpath)} (placeholder)")
        return False

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {"kv_cache": "darkblue", "no_cache": "darkorange"}
    markers = {"kv_cache": "o", "no_cache": "s"}
    extrap_colors = {"kv_cache": "crimson", "no_cache": "purple"}

    for mode in ["kv_cache", "no_cache"]:
        mode_df = df[df["mode"] == mode].copy()
        if mode_df.empty:
            print(f"\n⚠️ WARNING: No rows for mode={mode} in {filename_stem} plot.")
            continue

        agg = mode_df.groupby(x_col, as_index=False).agg(
            {speedup_col: ["mean", "std", "count"]}
        )
        agg.columns = [x_col, "speedup_mean", "speedup_std", "n"]

        x_obs = agg[x_col].to_numpy(dtype=float)
        y_obs = agg["speedup_mean"].to_numpy(dtype=float)
        y_std = agg["speedup_std"].fillna(0).to_numpy(dtype=float)

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
            label=f"{mode} (observed)",
            zorder=5,
        )

        x_log, y_log, params_log, r2_log = fit_and_extrapolate(
            x_obs, y_obs, max_x, model="log"
        )
        x_pow, y_pow, params_pow, r2_pow = fit_and_extrapolate(
            x_obs, y_obs, max_x, model="power"
        )

        if r2_log is not None and r2_pow is not None:
            if r2_log >= r2_pow:
                x_extrap, y_extrap, r2, model_name = x_log, y_log, r2_log, "log"
            else:
                x_extrap, y_extrap, r2, model_name = x_pow, y_pow, r2_pow, "power"
        elif r2_log is not None:
            x_extrap, y_extrap, r2, model_name = x_log, y_log, r2_log, "log"
        elif r2_pow is not None:
            x_extrap, y_extrap, r2, model_name = x_pow, y_pow, r2_pow, "power"
        else:
            x_extrap, y_extrap, r2, model_name = None, None, None, "none"

        if x_extrap is not None:
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
                {x_col: x_extrap, f"speedup_{mode}_predicted": y_extrap}
            )
            obs_df = pd.DataFrame({x_col: x_obs, f"speedup_{mode}_observed": y_obs})
            extrap_df = extrap_df.merge(obs_df, on=x_col, how="left")
            extrap_path = os.path.join(
                outdir, f"{filename_stem}_extrapolation_{mode}.csv"
            )
            extrap_df.to_csv(extrap_path, index=False)
            print(f"  → {os.path.basename(extrap_path)}")

    ax.axhline(
        1.0,
        color="black",
        linewidth=2,
        linestyle=":",
        alpha=0.7,
        label="Break-even (1.0)",
    )
    ax.set_xscale("log")
    ax.set_xlabel(f"{x_label} (log scale)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Tau speedup (higher = tau faster)", fontsize=14, fontweight="bold")

    ax.set_title(
        f"Tau Speedup vs {x_label}\nkv_cache vs no_cache • Extrapolated to {max_x:,}",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, which="both", linestyle="--")

    textstr = "kv_cache = inference decode with cache\nno_cache = training-like decode without cache"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.6)
    ax.text(
        0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10, va="top", bbox=props
    )

    outpath = os.path.join(outdir, f"tau_speedup_vs_{filename_stem}_bothmodes.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"  → {os.path.basename(outpath)}")
    return True


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Tau speedup extrapolation (gen_tokens / prompt_len / nembd)"
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
    ap.add_argument(
        "--max_nembd", type=int, default=32_768, help="Extrapolate nembd to this value"
    )

    # nembd controls
    ap.add_argument(
        "--base_nembd",
        type=int,
        default=384,
        help="Base embedding dimension (for embd_mult join)",
    )
    ap.add_argument(
        "--nembd_source",
        choices=["prompt_len", "join_embd_mult"],
        default="prompt_len",
        help="How to create nembd if missing. Default uses prompt_len directly.",
    )
    ap.add_argument(
        "--token_latencies_csv",
        default=None,
        help="token_latencies.csv containing embd_mult (required for join_embd_mult)",
    )
    ap.add_argument(
        "--runs_csv",
        default=None,
        help="runs.csv containing run_id->variant mapping (required for join_embd_mult)",
    )

    args = ap.parse_args()

    ensure_dir(args.outdir)

    print(f"\nLoading {args.pairwise_csv}...")
    if not os.path.exists(args.pairwise_csv):
        die(f"❌ ERROR: pairwise_csv not found: {args.pairwise_csv}")

    df = pd.read_csv(args.pairwise_csv)
    print(f"✓ Loaded {len(df)} pairwise speedup rows")

    # Basic required columns for any plot
    for c in ["mode", "gen_tokens", "prompt_len"]:
        if c not in df.columns:
            die(
                f"❌ ERROR: pairwise_csv missing required column '{c}'. Found: {df.columns.tolist()}"
            )

    speedup_col = choose_speedup_col(df)

    # Ensure nembd exists (default: prompt_len -> nembd)
    df = ensure_nembd(df, args)

    # Print dataset summary
    print("\nDataset Summary:")
    print(f"  Speedup column: {speedup_col}")
    print(f"  Available columns: {df.columns.tolist()}")
    print(f"  Modes: {df['mode'].unique()}")
    print(f"  gen_tokens unique values: {sorted(df['gen_tokens'].unique().tolist())}")
    print(f"  prompt_len unique values: {sorted(df['prompt_len'].unique().tolist())}")
    if "nembd" in df.columns:
        nembd_vals = df["nembd"].dropna().unique().tolist()
        print(f"  nembd unique values: {sorted([int(v) for v in nembd_vals])}")
    else:
        print("  nembd: COLUMN NOT FOUND ⚠️ (a placeholder plot will be generated)")

    print("\nGenerating extrapolation plots (kv_cache & no_cache)...\n")

    print("[1/3] speedup vs gen_tokens ...")
    plot_speedup_vs_dimension(
        df=df,
        outdir=args.outdir,
        speedup_col=speedup_col,
        x_col="gen_tokens",
        x_label="Generation Length (gen_tokens)",
        filename_stem="gentokens",
        max_x=args.max_gen_tokens,
    )

    print("\n[2/3] speedup vs prompt_len ...")
    plot_speedup_vs_dimension(
        df=df,
        outdir=args.outdir,
        speedup_col=speedup_col,
        x_col="prompt_len",
        x_label="Prompt Length (prompt_len)",
        filename_stem="promptlen",
        max_x=args.max_prompt_len,
    )

    print("\n[3/3] speedup vs nembd ...")
    if "nembd" in df.columns:
        df_nembd = df.dropna(subset=["nembd"]).copy()
        if len(df_nembd) == 0:
            print(
                "\n❌ ERROR: nembd exists but all values are NaN; generating placeholder plot."
            )
    else:
        df_nembd = df.iloc[0:0].copy()

    plot_speedup_vs_dimension(
        df=df_nembd if len(df_nembd) else df,
        outdir=args.outdir,
        speedup_col=speedup_col,
        x_col="nembd",
        x_label="Embedding Dimension (nembd)",
        filename_stem="nembd",
        max_x=args.max_nembd,
    )

    print(f"\n✓ All outputs written to: {args.outdir}/")
    print("Key files:")
    print(" - tau_speedup_vs_gentokens_bothmodes.png")
    print(" - tau_speedup_vs_promptlen_bothmodes.png")
    print(" - tau_speedup_vs_nembd_bothmodes.png")
    print(" - gentokens_extrapolation_{kv_cache,no_cache}.csv")
    print(" - promptlen_extrapolation_{kv_cache,no_cache}.csv")
    print(" - nembd_extrapolation_{kv_cache,no_cache}.csv (if nembd has variation)")


if __name__ == "__main__":
    main()
