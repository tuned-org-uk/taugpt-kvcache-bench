# taugpt-kvcache-bench

Faster Domain-driven KV-cache for the transformer architecture thanks to taumode (`arrowspace-rs`).

## Assumptions of comparison

* "no_cache" mode is full forward / training-like proxy (dense dot-product vs lambda-distance) without mixing in cache behavior.
* "kv_cache" mode is a clean inference-like proxy with three phases:
  * `prefill_ms` = build cache from prompt (decode-priming)
  * `ttft_ms` = first generated token
  * `decode_total_ms` = steady-state decode loop
* `end_to_end_ms` gives a single number to optimise for serving, while still letting inspect phase breakdowns.

## Build
```
# for CPU
cargo build --release
# or
cargo build --release --features wgpu
# or
cargo build --release --features cuda
```

## Usage
* Run both NanoGPT and TauGPT: multi-head "manifold native" (check defaults in `main.rs`)
```
RUST_LOG=info cargo run --release --   --prompts ./prompts.jsonl   --gen-tokens 2048
```


## Run analysis


```
# aggregate results
$ python analysis/aggregation_analysis.py --root output/ --out report/aggregation
# plot
$ python analysis/all_tokens_latencies.py   --tokencsv report/aggregation/tokenlatencies__normalized.csv   --outdir report/token_latencies 
# extrapolate
$ python analysis/extrapolate_tau_speedup.py   --pairwise_csv report/aggregation/tau_vs_nano__pairwise_speedups.csv   --outdir plots_extrapolation   --max_gen_tokens 1000000   --max_prompt_len 10000000
```
