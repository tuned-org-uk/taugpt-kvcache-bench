#!/usr/bin/env python3
"""
aggregation_analysis.py

Complete script for generating aggregated statistics from benchmark runs.
Supports directories with runs.csv and tokenlatencies.csv files.

METRIC INTERPRETATIONS:
  - token_ms: milliseconds per token (LOWER is better)
  - tokens_per_sec: tokens per second (HIGHER is better)
  - *_ms metrics: latency in milliseconds (LOWER is better)
  - rss_*_mb: memory in MB (LOWER is better)

IMPORTANT NOTES:
  - TTFT is measured differently in kv_cache vs no_cache modes
  - kv_cache: ttft_ms = decode step for first token (after prefill)
  - no_cache: ttft_ms = full forward pass through prompt context
  - DO NOT compare TTFT across cache modes

Usage:
    python aggregation_analysis.py --root ./benchmarks --out ./results
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

RUNS_FILE = "runs.csv"
TOK_FILES = ["tokenlatencies.csv", "token_latencies.csv"]

# Canonical column names
CANON_RUNS_NUMERIC = [
    "prompt_len",
    "gen_tokens",
    "prefill_ms",
    "prime_ms",
    "ttft_ms",
    "decode_total_ms",
    "tokens_per_sec",
    "p50_token_ms",
    "p95_token_ms",
    "rss_before_mb",
    "rss_after_mb",
]
CANON_TOK_NUMERIC = ["token_index", "token_ms"]

# Column rename mappings
RUNS_RENAMES: Dict[str, str] = {
    "runid": "run_id",
    "promptid": "prompt_id",
    "promptlen": "prompt_len",
    "gentokens": "gen_tokens",
    "prefillms": "prefill_ms",
    "primems": "prime_ms",
    "ttftms": "ttft_ms",
    "decodetotalms": "decode_total_ms",
    "tokenspersec": "tokens_per_sec",
    "p50tokenms": "p50_token_ms",
    "p95tokenms": "p95_token_ms",
    "rssbeforemb": "rss_before_mb",
    "rssaftermb": "rss_after_mb",
}

TOK_RENAMES: Dict[str, str] = {
    "runid": "run_id",
    "promptid": "prompt_id",
    "tokenindex": "token_index",
    "tokenms": "token_ms",
}

# Relative position bins for token analysis
POSBINS = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0000001])
POSBINLABELS = ["0-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-100%"]


@dataclass(frozen=True)
class Experiment:
    name: str
    path: Path
    runs_path: Path
    tok_path: Path
    device: str
    variant: str
    timestamp: Optional[int]


def _infer_device_and_variant(dirname: str) -> Tuple[str, str, Optional[int]]:
    """Extract device, variant, and timestamp from directory name."""
    ts = None
    m = re.match(r"^(\d+)_?(.*)$", dirname)
    rest = dirname
    if m:
        try:
            ts = int(m.group(1))
        except Exception:
            ts = None
        rest = m.group(2) if m.group(2) else dirname

    lower = dirname.lower()
    device = "unknown"
    if "cpu" in lower:
        device = "cpu"
    elif "wgpu" in lower:
        device = "wgpu"
    elif "cuda" in lower:
        device = "cuda"
    elif "mps" in lower:
        device = "mps"

    variant = rest.strip("_") if isinstance(rest, str) else "unknown"
    if not variant:
        variant = "default"
    return device, variant, ts


def _find_tok_file(dirpath: Path) -> Optional[Path]:
    """Find token latencies file."""
    for name in TOK_FILES:
        p = dirpath / name
        if p.exists():
            return p
    return None


def discover_experiments(root: Path) -> List[Experiment]:
    """Discover all experiment directories containing runs.csv and token latencies."""
    root = root.expanduser().resolve()
    exps: List[Experiment] = []

    # Check if root itself is an experiment directory
    tok_p = _find_tok_file(root)
    if (root / RUNS_FILE).exists() and tok_p is not None:
        device, variant, ts = _infer_device_and_variant(root.name)
        exps.append(
            Experiment(
                name=root.name,
                path=root,
                runs_path=root / RUNS_FILE,
                tok_path=tok_p,
                device=device,
                variant=variant,
                timestamp=ts,
            )
        )
        return exps

    # Check subdirectories
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        runs_p = p / RUNS_FILE
        tok_p = _find_tok_file(p)
        if runs_p.exists() and tok_p is not None:
            device, variant, ts = _infer_device_and_variant(p.name)
            exps.append(
                Experiment(
                    name=p.name,
                    path=p,
                    runs_path=runs_p,
                    tok_path=tok_p,
                    device=device,
                    variant=variant,
                    timestamp=ts,
                )
            )
    return exps


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Convert specified columns to numeric."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _normalize_modes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize mode column values."""
    if "mode" not in df.columns:
        return df
    m = df["mode"].astype(str).str.lower()
    m = m.replace({"nocache": "no_cache", "kvcache": "kv_cache"})
    df["mode"] = m
    return df


def _normalize_engines(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize engine column values."""
    if "engine" not in df.columns:
        return df
    df["engine"] = (
        df["engine"]
        .astype(str)
        .str.lower()
        .replace({"nanogpt": "nano", "taugpt": "tau"})
    )
    return df


def _rename_columns(df: pd.DataFrame, renames: Dict[str, str]) -> pd.DataFrame:
    """Rename columns to canonical names."""
    lower_map = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=lower_map)
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
    return df


def load_runs(exps: List[Experiment]) -> pd.DataFrame:
    """Load and normalize all runs data."""
    dfs = []
    for e in exps:
        d = pd.read_csv(e.runs_path)
        d = _rename_columns(d, RUNS_RENAMES)
        d = _normalize_engines(d)
        d = _normalize_modes(d)
        d["experiment"] = e.name
        d["device"] = e.device
        d["variant"] = e.variant
        d["timestamp"] = e.timestamp
        d = _coerce_numeric(d, CANON_RUNS_NUMERIC)
        dfs.append(d)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_token_latencies(exps: List[Experiment]) -> pd.DataFrame:
    """Load and normalize all token latency data."""
    dfs = []
    for e in exps:
        d = pd.read_csv(e.tok_path)
        d = _rename_columns(d, TOK_RENAMES)
        d = _normalize_engines(d)
        d = _normalize_modes(d)
        d["experiment"] = e.name
        d["device"] = e.device
        d["variant"] = e.variant
        d["timestamp"] = e.timestamp
        d = _coerce_numeric(d, CANON_TOK_NUMERIC)
        dfs.append(d)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def add_derived_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """Add computed metrics to runs data."""
    d = runs.copy()

    # Total time (prefill + decode, excluding TTFT to avoid mode-specific semantics)
    if "prefill_ms" in d.columns and "decode_total_ms" in d.columns:
        d["total_ms_no_ttft"] = d["prefill_ms"].fillna(0.0) + d[
            "decode_total_ms"
        ].fillna(0.0)

    # For backward compatibility, keep total_ms including ttft (but don't compare across modes)
    if (
        "prefill_ms" in d.columns
        and "decode_total_ms" in d.columns
        and "ttft_ms" in d.columns
    ):
        d["total_ms"] = (
            d["prefill_ms"].fillna(0.0)
            + d["ttft_ms"].fillna(0.0)
            + d["decode_total_ms"].fillna(0.0)
        )

    # Memory delta
    if "rss_before_mb" in d.columns and "rss_after_mb" in d.columns:
        d["rss_delta_mb"] = d["rss_after_mb"] - d["rss_before_mb"]

    return d


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate runs by grouping variables."""
    if runs.empty:
        return runs

    group_cols = [
        c
        for c in ["device", "variant", "engine", "mode", "prompt_len", "gen_tokens"]
        if c in runs.columns
    ]
    metric_cols = [
        c
        for c in [
            "prefill_ms",
            "prime_ms",
            "ttft_ms",
            "decode_total_ms",
            "tokens_per_sec",
            "p50_token_ms",
            "p95_token_ms",
            "rss_before_mb",
            "rss_after_mb",
            "rss_delta_mb",
            "total_ms",
            "total_ms_no_ttft",
        ]
        if c in runs.columns
    ]

    agg = (
        runs.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Flatten multi-level column names
    flat_cols = []
    for c in agg.columns:
        if isinstance(c, tuple):
            flat_cols.append(c[0] if c[1] == "" else f"{c[0]}__{c[1]}")
        else:
            flat_cols.append(c)
    agg.columns = flat_cols
    return agg


def compute_pairwise_speedups(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute speedups between tau and nano engines.

    Speedup calculation:
    - For throughput metrics (higher is better): speedup = tau / nano
    - For latency metrics (lower is better): speedup = nano / tau
    - For memory metrics: ratio = tau / nano, saving = 1 - ratio

    A speedup > 1.0 means tau is faster.
    """
    if agg.empty or "engine" not in agg.columns:
        return pd.DataFrame()

    keys = [
        c
        for c in ["device", "variant", "mode", "prompt_len", "gen_tokens"]
        if c in agg.columns
    ]

    def col(m: str) -> str:
        return f"{m}__mean"

    # Throughput: higher is better
    metrics_thr = [m for m in ["tokens_per_sec"] if col(m) in agg.columns]

    # Latency: lower is better
    metrics_lat = [
        m
        for m in [
            "p50_token_ms",
            "p95_token_ms",
            "decode_total_ms",
            "total_ms_no_ttft",
            "token_ms",
        ]
        if col(m) in agg.columns
    ]

    # Memory: lower is better
    metrics_mem = [m for m in ["rss_delta_mb", "rss_after_mb"] if col(m) in agg.columns]

    tau = agg[agg["engine"].astype(str).str.lower().isin(["tau", "taugpt"])].copy()
    nano = agg[agg["engine"].astype(str).str.lower().isin(["nano", "nanogpt"])].copy()

    if tau.empty or nano.empty:
        return pd.DataFrame()

    tau = tau[keys + [col(m) for m in metrics_thr + metrics_lat + metrics_mem]]
    nano = nano[keys + [col(m) for m in metrics_thr + metrics_lat + metrics_mem]]
    merged = tau.merge(nano, on=keys, how="inner", suffixes=("_tau", "_nano"))

    out = merged[keys].copy()

    # Throughput speedups: tau/nano
    for m in metrics_thr:
        out[f"{m}_speedup_tau_over_nano"] = (
            merged[f"{col(m)}_tau"] / merged[f"{col(m)}_nano"]
        )

    # Latency speedups: nano/tau (>1 means tau is faster)
    for m in metrics_lat:
        out[f"{m}_speedup_tau_over_nano"] = (
            merged[f"{col(m)}_nano"] / merged[f"{col(m)}_tau"]
        )

    # Memory ratios and savings
    for m in metrics_mem:
        out[f"{m}_ratio_tau_over_nano"] = (
            merged[f"{col(m)}_tau"] / merged[f"{col(m)}_nano"]
        )
        out[f"{m}_saving_frac_tau_vs_nano"] = 1.0 - out[f"{m}_ratio_tau_over_nano"]

    return out


def compute_cache_mode_gains(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute performance gains from KV cache vs no cache.

    IMPORTANT: TTFT is excluded because it has different semantics:
    - kv_cache: ttft_ms = decode step for first token
    - no_cache: ttft_ms = full forward pass

    Gain calculation:
    - For throughput (higher is better): gain = kv_cache / no_cache
    - For latency (lower is better): gain = no_cache / kv_cache
    - For memory: ratio = kv_cache / no_cache, overhead = ratio - 1

    A gain > 1.0 means kv_cache is faster.
    """
    if agg.empty or "mode" not in agg.columns:
        return pd.DataFrame()

    keys = [
        c
        for c in ["device", "variant", "engine", "prompt_len", "gen_tokens"]
        if c in agg.columns
    ]

    def col(m: str) -> str:
        return f"{m}__mean"

    # Throughput: higher is better
    metrics_thr = [m for m in ["tokens_per_sec"] if col(m) in agg.columns]

    # Latency: lower is better (EXCLUDE ttft_ms - semantics differ across modes)
    metrics_lat = [
        m
        for m in [
            "p50_token_ms",
            "p95_token_ms",
            "decode_total_ms",
            "total_ms_no_ttft",
            "token_ms",
        ]
        if col(m) in agg.columns
    ]

    # Memory: lower is better
    metrics_mem = [m for m in ["rss_delta_mb", "rss_after_mb"] if col(m) in agg.columns]

    kv = agg[agg["mode"].astype(str).str.lower() == "kv_cache"][
        keys + [col(m) for m in metrics_thr + metrics_lat + metrics_mem]
    ].copy()
    nc = agg[agg["mode"].astype(str).str.lower() == "no_cache"][
        keys + [col(m) for m in metrics_thr + metrics_lat + metrics_mem]
    ].copy()

    if kv.empty or nc.empty:
        return pd.DataFrame()

    merged = kv.merge(nc, on=keys, how="inner", suffixes=("_kv", "_nc"))
    out = merged[keys].copy()

    # Throughput gains: kv/nc
    for m in metrics_thr:
        out[f"{m}_gain_kv_over_no_cache"] = (
            merged[f"{col(m)}_kv"] / merged[f"{col(m)}_nc"]
        )

    # Latency gains: nc/kv (>1 means kv_cache is faster)
    for m in metrics_lat:
        out[f"{m}_gain_kv_over_no_cache"] = (
            merged[f"{col(m)}_nc"] / merged[f"{col(m)}_kv"]
        )

    # Memory overhead
    for m in metrics_mem:
        out[f"{m}_ratio_kv_over_no_cache"] = (
            merged[f"{col(m)}_kv"] / merged[f"{col(m)}_nc"]
        )
        out[f"{m}_increase_frac_kv_vs_no_cache"] = (
            out[f"{m}_ratio_kv_over_no_cache"] - 1.0
        )

    return out


def infer_gen_tokens_from_tok(tok_lat: pd.DataFrame) -> pd.DataFrame:
    """Infer gen_tokens per run from max(token_index)+1."""
    if tok_lat.empty or "token_index" not in tok_lat.columns:
        return tok_lat

    d = tok_lat.copy()
    d["token_index"] = pd.to_numeric(d["token_index"], errors="coerce")

    # Use only columns that exist
    wanted = ["run_id", "prompt_id", "engine", "mode", "experiment"]
    keys = [k for k in wanted if k in d.columns]
    if not keys:
        print("Warning: cannot infer gen_tokens, no key columns found")
        return d

    base = d.dropna(subset=keys + ["token_index"]).copy()
    if base.empty:
        return d

    g = (
        base.groupby(keys, dropna=False, as_index=False)["token_index"]
        .max()
        .rename(columns={"token_index": "token_index_max"})
    )
    g["gen_tokens"] = (g["token_index_max"] + 1.0).round().astype("Int64")
    g = g.drop(columns=["token_index_max"])

    before = len(d)
    d = d.merge(g, on=keys, how="left")
    d = d.dropna(subset=["gen_tokens"]).copy()
    after = len(d)
    if after < before:
        print(f"Warning: dropped {before - after} token rows with missing gen_tokens")

    d["gen_tokens"] = d["gen_tokens"].astype("int64")
    return d


def add_relpos_bins(tok_lat: pd.DataFrame) -> pd.DataFrame:
    """Add relative position bins."""
    if tok_lat.empty:
        return tok_lat
    if "gen_tokens" not in tok_lat.columns or "token_index" not in tok_lat.columns:
        return tok_lat

    d = tok_lat.copy()
    gt = pd.to_numeric(d["gen_tokens"], errors="coerce")
    ti = pd.to_numeric(d["token_index"], errors="coerce")
    d["relpos"] = ti / gt
    d["relpos_bin"] = pd.cut(
        d["relpos"].clip(0, 1),
        bins=POSBINS,
        labels=POSBINLABELS,
        include_lowest=True,
        right=True,
    )
    return d


def aggregate_token_latencies(tok_lat: pd.DataFrame) -> pd.DataFrame:
    """Aggregate token latencies by device/variant/engine/mode."""
    if tok_lat.empty:
        return tok_lat

    group_cols = [
        c
        for c in ["device", "variant", "engine", "mode", "experiment"]
        if c in tok_lat.columns
    ]

    if not group_cols:
        return pd.DataFrame()

    agg = (
        tok_lat.groupby(group_cols, dropna=False)["token_ms"]
        .agg(
            [
                "mean",
                "std",
                "min",
                ("p25", lambda x: x.quantile(0.25)),
                ("p50", lambda x: x.quantile(0.50)),
                ("p75", lambda x: x.quantile(0.75)),
                ("p95", lambda x: x.quantile(0.95)),
                "max",
                "count",
            ]
        )
        .reset_index()
    )

    agg.columns = [f"token_ms__{c}" if c not in group_cols else c for c in agg.columns]
    return agg


def aggregate_token_latencies_by_posbin(tok_lat: pd.DataFrame) -> pd.DataFrame:
    """Aggregate token latencies grouped by relative position bin."""
    if tok_lat.empty or "relpos_bin" not in tok_lat.columns:
        return pd.DataFrame()

    group_cols = [
        c
        for c in ["device", "variant", "engine", "mode", "relpos_bin"]
        if c in tok_lat.columns
    ]

    if not group_cols:
        return pd.DataFrame()

    agg = (
        tok_lat.groupby(group_cols, dropna=False)["token_ms"]
        .agg(
            [
                "mean",
                "std",
                "min",
                ("p50", lambda x: x.quantile(0.50)),
                ("p95", lambda x: x.quantile(0.95)),
                "max",
                "count",
            ]
        )
        .reset_index()
    )

    agg.columns = [f"token_ms__{c}" if c not in group_cols else c for c in agg.columns]
    return agg


def compute_pairwise_token_speedups(
    tok_agg: pd.DataFrame, extra_keys: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compute tau vs nano speedups for token latencies."""
    if tok_agg.empty or "engine" not in tok_agg.columns:
        return pd.DataFrame()

    keys = [c for c in ["device", "variant", "mode"] if c in tok_agg.columns]
    if extra_keys:
        keys += [k for k in extra_keys if k in tok_agg.columns]

    def col(stat: str) -> str:
        return f"token_ms__{stat}"

    metrics = [s for s in ["mean", "p50", "p95"] if col(s) in tok_agg.columns]
    if not metrics:
        return pd.DataFrame()

    tau = tok_agg[
        tok_agg["engine"].astype(str).str.lower().isin(["tau", "taugpt"])
    ].copy()
    nano = tok_agg[
        tok_agg["engine"].astype(str).str.lower().isin(["nano", "nanogpt"])
    ].copy()

    if tau.empty or nano.empty:
        return pd.DataFrame()

    tau = tau[keys + [col(m) for m in metrics]]
    nano = nano[keys + [col(m) for m in metrics]]
    merged = tau.merge(nano, on=keys, how="inner", suffixes=("_tau", "_nano"))

    out = merged[keys].copy()
    for m in metrics:
        # Latency: nano/tau (>1 means tau is faster)
        out[f"token_ms_{m}_speedup_tau_over_nano"] = (
            merged[f"{col(m)}_nano"] / merged[f"{col(m)}_tau"]
        )
    return out


def compute_cache_mode_token_gains(
    tok_agg: pd.DataFrame, extra_keys: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compute kv_cache vs no_cache gains for token latencies."""
    if tok_agg.empty or "mode" not in tok_agg.columns:
        return pd.DataFrame()

    keys = [c for c in ["device", "variant", "engine"] if c in tok_agg.columns]
    if extra_keys:
        keys += [k for k in extra_keys if k in tok_agg.columns]

    def col(stat: str) -> str:
        return f"token_ms__{stat}"

    metrics = [s for s in ["mean", "p50", "p95"] if col(s) in tok_agg.columns]
    if not metrics:
        return pd.DataFrame()

    kv = tok_agg[tok_agg["mode"].astype(str).str.lower() == "kv_cache"][
        keys + [col(m) for m in metrics]
    ].copy()
    nc = tok_agg[tok_agg["mode"].astype(str).str.lower() == "no_cache"][
        keys + [col(m) for m in metrics]
    ].copy()

    if kv.empty or nc.empty:
        return pd.DataFrame()

    merged = kv.merge(nc, on=keys, how="inner", suffixes=("_kv", "_nc"))
    out = merged[keys].copy()
    for m in metrics:
        # Latency: nc/kv (>1 means kv_cache is faster)
        out[f"token_ms_{m}_gain_kv_over_no_cache"] = (
            merged[f"{col(m)}_nc"] / merged[f"{col(m)}_kv"]
        )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate aggregated benchmark statistics"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output directory for aggregated results"
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering experiments in {root}...")
    exps = discover_experiments(root)
    print(f"Found {len(exps)} experiments:")
    for e in exps:
        print(f"  - {e.name} (device={e.device}, variant={e.variant})")

    if not exps:
        print("No experiments found. Exiting.")
        return

    print("\nLoading runs data...")
    runs = load_runs(exps)
    print(f"Loaded {len(runs)} runs")

    print("\nLoading token latencies...")
    tok_lat = load_token_latencies(exps)
    print(f"Loaded {len(tok_lat)} token latency records")

    print("\nAdding derived metrics...")
    runs = add_derived_metrics(runs)

    print("\nAggregating runs...")
    agg = aggregate_runs(runs)
    print(f"Aggregated into {len(agg)} groups")

    print("\nInferring gen_tokens and adding relpos bins...")
    tok_lat2 = infer_gen_tokens_from_tok(tok_lat)
    tok_lat2 = add_relpos_bins(tok_lat2)

    print("\nAggregating token latencies (overall)...")
    tok_agg = aggregate_token_latencies(tok_lat2)
    print(f"Aggregated token latencies into {len(tok_agg)} groups")

    print("\nAggregating token latencies (by position bin)...")
    tok_agg_posbin = aggregate_token_latencies_by_posbin(tok_lat2)
    print(f"Aggregated token latencies by posbin into {len(tok_agg_posbin)} groups")

    print("\nComputing pairwise speedups (tau vs nano, runs)...")
    speedups = compute_pairwise_speedups(agg)
    print(f"Computed {len(speedups)} speedup comparisons")

    print("\nComputing cache mode gains (runs)...")
    cache_gains = compute_cache_mode_gains(agg)
    print(f"Computed {len(cache_gains)} cache comparisons")

    print("\nComputing token speedups (tau vs nano)...")
    tok_speedups = compute_pairwise_token_speedups(tok_agg)
    tok_speedups_posbin = compute_pairwise_token_speedups(
        tok_agg_posbin, extra_keys=["relpos_bin"]
    )
    print(f"Computed {len(tok_speedups)} overall, {len(tok_speedups_posbin)} by posbin")

    print("\nComputing token cache gains...")
    tok_cache_gains = compute_cache_mode_token_gains(tok_agg)
    tok_cache_gains_posbin = compute_cache_mode_token_gains(
        tok_agg_posbin, extra_keys=["relpos_bin"]
    )
    print(
        f"Computed {len(tok_cache_gains)} overall, {len(tok_cache_gains_posbin)} by posbin"
    )

    # Save all outputs
    print("\nSaving results...")
    runs.to_csv(out_dir / "runs__normalized.csv", index=False)
    print(f"  - runs__normalized.csv ({len(runs)} rows)")

    tok_lat2.to_csv(out_dir / "tokenlatencies__normalized.csv", index=False)
    print(f"  - tokenlatencies__normalized.csv ({len(tok_lat2)} rows)")

    agg.to_csv(out_dir / "runs__agg.csv", index=False)
    print(f"  - runs__agg.csv ({len(agg)} rows)")

    if not tok_agg.empty:
        tok_agg.to_csv(out_dir / "tokenlatencies__agg.csv", index=False)
        print(f"  - tokenlatencies__agg.csv ({len(tok_agg)} rows)")

    if not tok_agg_posbin.empty:
        tok_agg_posbin.to_csv(
            out_dir / "tokenlatencies__agg_by_posbin.csv", index=False
        )
        print(f"  - tokenlatencies__agg_by_posbin.csv ({len(tok_agg_posbin)} rows)")

    if not speedups.empty:
        speedups.to_csv(out_dir / "tau_vs_nano__pairwise_speedups.csv", index=False)
        print(f"  - tau_vs_nano__pairwise_speedups.csv ({len(speedups)} rows)")

    if not cache_gains.empty:
        cache_gains.to_csv(out_dir / "cache_mode__gains.csv", index=False)
        print(f"  - cache_mode__gains.csv ({len(cache_gains)} rows)")

    if not tok_speedups.empty:
        tok_speedups.to_csv(out_dir / "tau_vs_nano__token_speedups.csv", index=False)
        print(f"  - tau_vs_nano__token_speedups.csv ({len(tok_speedups)} rows)")

    if not tok_speedups_posbin.empty:
        tok_speedups_posbin.to_csv(
            out_dir / "tau_vs_nano__token_speedups_by_posbin.csv", index=False
        )
        print(
            f"  - tau_vs_nano__token_speedups_by_posbin.csv ({len(tok_speedups_posbin)} rows)"
        )

    if not tok_cache_gains.empty:
        tok_cache_gains.to_csv(out_dir / "cache_mode__token_gains.csv", index=False)
        print(f"  - cache_mode__token_gains.csv ({len(tok_cache_gains)} rows)")

    if not tok_cache_gains_posbin.empty:
        tok_cache_gains_posbin.to_csv(
            out_dir / "cache_mode__token_gains_by_posbin.csv", index=False
        )
        print(
            f"  - cache_mode__token_gains_by_posbin.csv ({len(tok_cache_gains_posbin)} rows)"
        )

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    if not agg.empty and "tokens_per_sec__mean" in agg.columns:
        print("\nThroughput (tokens/sec) - HIGHER is better:")
        print("-" * 70)
        summary = agg.groupby(["engine", "mode"])["tokens_per_sec__mean"].agg(
            ["mean", "min", "max"]
        )
        print(summary)

    if not tok_agg.empty and "token_ms__mean" in tok_agg.columns:
        print("\nPer-token latency (ms) - LOWER is better:")
        print("-" * 70)
        tok_summary = tok_agg.groupby(["engine", "mode"])[
            ["token_ms__mean", "token_ms__p50", "token_ms__p95"]
        ].mean()
        print(tok_summary)

    if not speedups.empty:
        print("\nSpeedup factors (tau over nano) - >1.0 means tau is faster:")
        print("-" * 70)
        speedup_cols = [c for c in speedups.columns if "speedup" in c]
        for col in speedup_cols[:5]:  # Show first 5
            mean_val = speedups[col].mean()
            print(f"  {col}: {mean_val:.3f}x")

    if not cache_gains.empty:
        print("\nCache gains (kv_cache over no_cache) - >1.0 means cache helps:")
        print("-" * 70)
        gain_cols = [c for c in cache_gains.columns if "gain" in c]
        for col in gain_cols[:5]:  # Show first 5
            mean_val = cache_gains[col].mean()
            print(f"  {col}: {mean_val:.3f}x")

    print(f"\n{'=' * 70}")
    print(f"All results saved to {out_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
