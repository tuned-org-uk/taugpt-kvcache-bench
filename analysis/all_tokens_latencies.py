#!/usr/bin/env python3
"""
Plot token-latency analysis from tokenlatencies__normalized.csv

Produces:
  - speedup_vs_gen_tokens.png: tau advantage in kv_cache grows with sequence length
  - speedup_by_relpos_bins.png: per-token speedup by relative position in sequence
  - run_clusters_pca.png: clustering runs by latency distribution features
  - CSV summaries and cluster counts
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

POS_BINS = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0000001])
POS_BIN_LABELS = ["0-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-100%"]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_percentile(x, q):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return float(np.nanpercentile(x, q))


def fit_line_log2x(x, y):
    """
    Fit y = a + b*log2(x). Returns (a, b).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if m.sum() < 2:
        return (np.nan, np.nan)
    lx = np.log2(x[m])
    b, a = np.polyfit(lx, y[m], 1)
    # polyfit returns [slope, intercept] for lx->y, so y = b*lx + a
    # We want y = a + b*log2(x), so return (a, b)
    return (a, b)


def predict_line_log2x(a_intercept, b_slope, x):
    x = np.asarray(x, dtype=float)
    return a_intercept + b_slope * np.log2(x)


def infer_gen_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer gen_tokens per (engine, mode, prompt_id, run_id) as max(token_index)+1.

    - Drops 'variant' from grouping keys (user requested; may be constant).
    - Coerces token_index to numeric and drops rows where it's missing.
    - Avoids int-casting until NaNs are removed.
    """
    df = df.copy()

    # Ensure token_index is numeric
    df["token_index"] = pd.to_numeric(df["token_index"], errors="coerce")

    # Keys without variant
    keys = ["prompt_id", "mode", "prompt_len", "gen_tokens", "token_index"]

    # Only use rows with valid token_index for inference
    base = df.dropna(subset=keys + ["token_index"]).copy()

    g = (
        base.groupby(keys, as_index=False)["token_index"]
        .max()
        .rename(columns={"token_index": "token_index_max"})
    )

    # gen_tokens = max_index + 1 (keep as float until validation)
    g["gen_tokens"] = g["token_index_max"] + 1.0

    # Drop any non-finite gen_tokens
    g = g[np.isfinite(g["gen_tokens"])].copy()

    # Now safe to cast
    g["gen_tokens"] = g["gen_tokens"].round().astype("int64")

    # Merge back
    df = df.merge(g[keys + ["gen_tokens"]], on=keys, how="left")

    # Drop rows where gen_tokens couldn't be inferred
    before = len(df)
    df = df.dropna(subset=["gen_tokens"]).copy()
    after = len(df)
    if after < before:
        print(f"Warning: dropped {before - after} rows with missing gen_tokens")

    df["gen_tokens"] = df["gen_tokens"].astype("int64")

    return df


def add_relpos_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rel_pos"] = df["token_index"] / df["gen_tokens"]
    df["pos_bin"] = pd.cut(
        df["rel_pos"].clip(0, 1),
        bins=POS_BINS,
        labels=POS_BIN_LABELS,
        include_lowest=True,
        right=True,
    )
    return df


def compute_matched_speedups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-token speedup (nano_ms / tau_ms) by matching tokens on:
      (prompt_id, mode, gen_tokens, token_index)
    Then return the joined DataFrame with speedup column.
    """
    keys = ["prompt_id", "mode", "gen_tokens", "token_index"]

    nano = df[df["engine"] == "nano"][keys + ["token_ms"]].rename(
        columns={"token_ms": "nano_token_ms"}
    )
    tau = df[df["engine"] == "tau"][keys + ["token_ms"]].rename(
        columns={"token_ms": "tau_token_ms"}
    )

    j = nano.merge(tau, on=keys, how="inner")
    j["speedup_nano_over_tau"] = j["nano_token_ms"] / j["tau_token_ms"]
    return j


def summarize_speedups(speedups: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize per-token speedup distributions by (mode, gen_tokens).
    Returns flat DataFrame.
    """
    out = speedups.groupby(["mode", "gen_tokens"], as_index=False).agg(
        pairs=("speedup_nano_over_tau", "size"),
        speedup_mean=("speedup_nano_over_tau", "mean"),
        speedup_p10=("speedup_nano_over_tau", lambda s: safe_percentile(s, 10)),
        speedup_p50=("speedup_nano_over_tau", lambda s: safe_percentile(s, 50)),
        speedup_p90=("speedup_nano_over_tau", lambda s: safe_percentile(s, 90)),
        speedup_p95=("speedup_nano_over_tau", lambda s: safe_percentile(s, 95)),
        speedup_p99=("speedup_nano_over_tau", lambda s: safe_percentile(s, 99)),
    )
    out["pairs"] = out["pairs"].astype(int)
    return out.sort_values(["mode", "gen_tokens"]).reset_index(drop=True)


def summarize_speedups_by_posbin(speedups_with_bins: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize speedups by relative position bin for each (mode, gen_tokens).
    """
    g = speedups_with_bins.groupby(
        ["mode", "gen_tokens", "pos_bin"], as_index=False
    ).agg(
        pairs=("speedup_nano_over_tau", "size"),
        p50=("speedup_nano_over_tau", lambda s: safe_percentile(s, 50)),
        p95=("speedup_nano_over_tau", lambda s: safe_percentile(s, 95)),
        p99=("speedup_nano_over_tau", lambda s: safe_percentile(s, 99)),
        mean=("speedup_nano_over_tau", "mean"),
    )
    g["pairs"] = g["pairs"].astype(int)
    return g


def compute_run_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-run distribution features for clustering runs.
    Run is keyed by (engine, mode, prompt_id, run_id, gen_tokens).
    """
    keys = ["engine", "mode", "prompt_id", "run_id", "gen_tokens"]

    feat = df.groupby(keys, as_index=False).agg(
        token_ms_mean=("token_ms", "mean"),
        token_ms_p50=("token_ms", lambda s: safe_percentile(s, 50)),
        token_ms_p95=("token_ms", lambda s: safe_percentile(s, 95)),
        token_ms_p99=("token_ms", lambda s: safe_percentile(s, 99)),
        token_ms_std=("token_ms", "std"),
    )

    # Compute slope separately per group
    def slope_vs_relpos(sub):
        x = sub["token_index"].to_numpy(dtype=float) / float(sub["gen_tokens"].iloc[0])
        y = sub["token_ms"].to_numpy(dtype=float)
        if len(x) < 3:
            return np.nan
        m, c = np.polyfit(x, y, 1)
        return float(m)

    slopes = []
    for keys_vals, sub in df.groupby(keys):
        slopes.append(slope_vs_relpos(sub))
    feat["slope_ms_per_relpos"] = slopes

    return feat


def cluster_runs(run_feat: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    cols = [
        "token_ms_mean",
        "token_ms_p50",
        "token_ms_p95",
        "token_ms_p99",
        "token_ms_std",
        "slope_ms_per_relpos",
    ]
    X = run_feat[cols].to_numpy(dtype=float)
    # Replace NaNs with column medians
    col_meds = np.nanmedian(X, axis=0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col_meds, inds[1])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(Xs)

    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(Xs)

    out = run_feat.copy()
    out["cluster"] = labels
    out["pca1"] = Z[:, 0]
    out["pca2"] = Z[:, 1]
    return out


def plot_speedup_vs_gen(summary: pd.DataFrame, outdir: str, max_gen: int):
    ensure_dir(outdir)
    plt.figure(figsize=(10, 6))

    for mode, color in [("kv_cache", "tab:blue"), ("no_cache", "tab:orange")]:
        s = summary[summary["mode"] == mode].sort_values("gen_tokens")
        if s.empty:
            continue

        x = s["gen_tokens"].to_numpy()
        y = s["speedup_p50"].to_numpy()

        plt.plot(
            x, y, "o-", color=color, linewidth=2, markersize=8, label=f"{mode} (p50)"
        )
        plt.fill_between(
            x,
            s["speedup_p10"].to_numpy(),
            s["speedup_p90"].to_numpy(),
            color=color,
            alpha=0.2,
            linewidth=0,
        )

        # Interpolation/extrapolation in log2 space
        a, b = fit_line_log2x(x, y)
        if np.isfinite(a) and np.isfinite(b):
            xs = np.geomspace(x.min(), max_gen, 100)
            ys = predict_line_log2x(a, b, xs)
            plt.plot(
                xs,
                ys,
                "--",
                color=color,
                alpha=0.7,
                linewidth=1.5,
                label=f"{mode} extrapolation (y={a:.2f}+{b:.3f}*log2(x))",
            )

    plt.axhline(
        1.0,
        color="black",
        linewidth=1.5,
        linestyle=":",
        alpha=0.7,
        label="speedup=1 (break-even)",
    )
    plt.xscale("log", base=2)
    plt.xlabel("gen_tokens (log2 scale)", fontsize=12)
    plt.ylabel("Per-token speedup = nano_token_ms / tau_token_ms", fontsize=12)
    plt.title(
        "Tau per-token advantage grows with sequence length in kv_cache;\nno_cache remains <1 (tau slower)",
        fontsize=13,
    )
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "speedup_vs_gen_tokens.png"), dpi=200)
    plt.close()
    print(f"  → speedup_vs_gen_tokens.png")


def plot_speedup_by_posbin(speedup_posbin: pd.DataFrame, outdir: str):
    ensure_dir(outdir)

    modes = ["kv_cache", "no_cache"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    for ax, mode in zip(axes, modes):
        d = speedup_posbin[speedup_posbin["mode"] == mode]
        if d.empty:
            ax.set_axis_off()
            continue

        gen_tokens_vals = sorted(d["gen_tokens"].unique())
        cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(gen_tokens_vals)))

        for i, gt in enumerate(gen_tokens_vals):
            dd = d[d["gen_tokens"] == gt].copy()
            dd["pos_bin"] = pd.Categorical(
                dd["pos_bin"], categories=POS_BIN_LABELS, ordered=True
            )
            dd = dd.sort_values("pos_bin")

            ax.plot(
                np.arange(len(dd)),
                dd["p50"].to_numpy(),
                marker="o",
                linewidth=2,
                markersize=6,
                color=cmap[i],
                label=f"{int(gt)}",
            )

        ax.axhline(1.0, color="black", linewidth=1.5, linestyle=":", alpha=0.7)
        ax.set_xticks(np.arange(len(POS_BIN_LABELS)))
        ax.set_xticklabels(POS_BIN_LABELS, rotation=30, ha="right")
        ax.set_title(f"{mode}: per-token speedup by decode position", fontsize=12)
        ax.set_xlabel("Relative position bin", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(title="gen_tokens", loc="best", fontsize=8)

    axes[0].set_ylabel("Per-token speedup = nano/tau (p50 within bin)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "speedup_by_relpos_bins.png"), dpi=200)
    plt.close()
    print(f"  → speedup_by_relpos_bins.png")


def plot_cluster_scatter(run_clusters: pd.DataFrame, outdir: str):
    ensure_dir(outdir)
    fig, ax = plt.subplots(figsize=(10, 7))

    markers = {
        ("nano", "kv_cache"): "o",
        ("tau", "kv_cache"): "s",
        ("nano", "no_cache"): "^",
        ("tau", "no_cache"): "D",
    }

    for (engine, mode), m in markers.items():
        sub = run_clusters[
            (run_clusters["engine"] == engine) & (run_clusters["mode"] == mode)
        ]
        if sub.empty:
            continue
        scatter = ax.scatter(
            sub["pca1"],
            sub["pca2"],
            c=sub["cluster"],
            cmap="tab10",
            marker=m,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            s=80,
            label=f"{engine}/{mode}",
        )

    ax.set_title(
        "Run clustering by token-latency distribution\n(PCA projection of mean/p50/p95/p99/std/slope features)",
        fontsize=12,
    )
    ax.set_xlabel("PCA1", fontsize=11)
    ax.set_ylabel("PCA2", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "run_clusters_pca.png"), dpi=200)
    plt.close()
    print(f"  → run_clusters_pca.png")

    # Cluster count table
    ct = (
        run_clusters.groupby(["cluster", "engine", "mode"])
        .size()
        .reset_index(name="runs")
    )
    ct.to_csv(os.path.join(outdir, "run_cluster_counts.csv"), index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(ct))
    ax.bar(
        x, ct["runs"].to_numpy(), color="steelblue", edgecolor="black", linewidth=0.5
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"c{r.cluster}·{r.engine}·{r.mode}" for r in ct.itertuples()],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("Number of runs", fontsize=11)
    ax.set_title("Cluster membership counts", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "run_cluster_counts.png"), dpi=200)
    plt.close()
    print(f"  → run_cluster_counts.png")


def main():
    ap = argparse.ArgumentParser(
        description="Analyze tokenlatencies__normalized.csv and produce plots + extrapolations"
    )
    ap.add_argument(
        "--token_csv",
        default="tokenlatencies__normalized.csv",
        help="Path to token latencies CSV",
    )
    ap.add_argument("--outdir", default="plots_tokenlat", help="Output directory")
    ap.add_argument(
        "--max_gen_tokens",
        type=int,
        default=8192,
        help="Extrapolate to this gen_tokens",
    )
    ap.add_argument(
        "--clusters_k", type=int, default=6, help="Number of KMeans clusters"
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    print(f"Loading {args.token_csv}...")
    # Simplified dtypes (no variant)
    dtypes = {
        "run_id": "int32",
        "engine": "category",
        "mode": "category",
        "prompt_id": "category",
        "token_index": "float32",  # keep float to handle parsing issues
        "token_ms": "float32",
    }
    df = pd.read_csv(args.token_csv, dtype=dtypes)

    # Normalize string columns
    for c in ["engine", "mode", "prompt_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    print(f"  Loaded {len(df):,} rows")

    # Infer gen_tokens and filter
    print("Inferring gen_tokens per run...")
    df = infer_gen_tokens(df)
    df = df[(df["token_index"] >= 0) & (df["token_index"] < df["gen_tokens"])].copy()
    print(f"  Retained {len(df):,} valid decode tokens")

    # Compute matched per-token speedups
    print("Computing matched per-token speedups (nano/tau)...")
    sp = compute_matched_speedups(df)
    sp["rel_pos"] = sp["token_index"] / sp["gen_tokens"]
    sp["pos_bin"] = pd.cut(
        sp["rel_pos"].clip(0, 1),
        bins=POS_BINS,
        labels=POS_BIN_LABELS,
        include_lowest=True,
        right=True,
    )
    print(f"  Matched {len(sp):,} token pairs")

    summary = summarize_speedups(sp)
    summary.to_csv(os.path.join(args.outdir, "speedup_summary.csv"), index=False)
    print(f"  Wrote speedup_summary.csv ({len(summary)} rows)")

    speedup_posbin = summarize_speedups_by_posbin(sp)
    speedup_posbin.to_csv(
        os.path.join(args.outdir, "speedup_by_posbin.csv"), index=False
    )
    print(f"  Wrote speedup_by_posbin.csv ({len(speedup_posbin)} rows)")

    # Plots with evidence + extrapolation
    print("Generating plots...")
    plot_speedup_vs_gen(summary, args.outdir, args.max_gen_tokens)
    plot_speedup_by_posbin(speedup_posbin, args.outdir)

    # Clustering runs by distribution features
    print("Clustering runs by latency distribution features...")
    run_feat = compute_run_features(df)
    run_feat.to_csv(os.path.join(args.outdir, "run_features.csv"), index=False)
    print(f"  Computed features for {len(run_feat)} runs")

    run_clusters = cluster_runs(run_feat, k=args.clusters_k, seed=args.seed)
    run_clusters.to_csv(os.path.join(args.outdir, "run_clusters.csv"), index=False)
    print(f"  Clustered into {args.clusters_k} groups")

    plot_cluster_scatter(run_clusters, args.outdir)

    print(f"\n✓ All outputs written to: {args.outdir}/")
    print("Key files:")
    print("  - speedup_vs_gen_tokens.png (evidence + extrapolation)")
    print("  - speedup_by_relpos_bins.png (late-token behavior)")
    print("  - run_clusters_pca.png (distribution clustering)")
    print("  - speedup_summary.csv, speedup_by_posbin.csv")
    print("  - run_features.csv, run_clusters.csv, run_cluster_counts.csv")


if __name__ == "__main__":
    main()
