* Run both NanoGPT and TauGPT: single-head "manifold native"
```
cargo run --release -- \
  --prompts ./prompts.jsonl \
  --max-new-tokens 128 \
  --repeats 5 \
  --warmup 1 \
  --runs-csv runs.csv \
  --token-csv token_latencies.csv \
  --run-tau \
  --manifold ./domain_manifold/manifold.parquet \
  --tau-mode median \
  --n-head 1 \
  --n-kv-head 1 \
  --n-layer 6
```

* Run same but "native multi-head"
--n-head 2 --n-kv-head 1
