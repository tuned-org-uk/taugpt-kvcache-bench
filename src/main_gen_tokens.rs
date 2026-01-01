use anyhow::{Result, anyhow};
use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};
use clap::Parser;
use csv::Writer;
use serde::Deserialize;
use std::fs::{File, create_dir_all};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{Pid, ProcessesToUpdate, System};

use tauformer::{
    backend::{AutoBackend, get_device, print_backend_info},
    causalattention::GptModel as NanoModel,
    config::NanoChatConfig,
    engine::GptCache,
    pretraining::parquet::{TauMode as ManifoldTauMode, load_domain_manifold},
    taugpt::{TauGptModel, TauKVCache},
};

type B = AutoBackend;

#[cfg(any(feature = "wgpu", feature = "cuda"))]
type TokenId = i32;
#[cfg(not(any(feature = "wgpu", feature = "cuda")))]
type TokenId = i64;

#[derive(Parser, Debug)]
struct Args {
    /// JSONL file: {"id": "...", "tokens": [1,2,3]}
    #[arg(long)]
    prompts: String,

    /// Output base directory. A timestamp subdir will be created inside it.
    #[arg(long, default_value = "output")]
    out_dir: String,

    /// Comma-separated prompt lengths to benchmark (tokens), e.g. "768,1152".
    /// If empty, defaults to 384*2 and 384*3.
    #[arg(long, default_value = "")]
    prompt_lens: String,

    /// Comma-separated gen token counts to benchmark, e.g. "1024,2048,4096,8000".
    #[arg(long, default_value = "1024,2048,4096,8000")]
    gen_tokens: String,

    /// Limit how many prompts from the JSONL to run (0 = all).
    #[arg(long, default_value_t = 3)]
    max_prompts: usize,

    /// Run TauGPT
    #[arg(long, default_value_t = true)]
    run_tau: bool,

    /// Run NanoGPT
    #[arg(long, default_value_t = true)]
    run_nano: bool,

    /// Path to manifold parquet for TauGPT sparse constructor
    #[arg(long, default_value = "./domain_manifold/manifold.parquet")]
    manifold: String,

    /// Tau mode ("median" | "mean")
    #[arg(long, default_value = "median")]
    tau_mode: String,

    /// Heads for NanoGPT config (keep n_embd = n_head * manifold_dim)
    #[arg(long, default_value_t = 1)]
    n_head: usize,

    /// KV heads for NanoGPT config
    #[arg(long, default_value_t = 1)]
    n_kv_head: usize,

    /// Layers for both models
    #[arg(long, default_value_t = 6)]
    n_layer: usize,
}

#[derive(Deserialize, Clone, Debug)]
struct PromptRow {
    id: String,
    tokens: Vec<TokenId>,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    NoCache,
    KvCache,
}

fn now_unix() -> u64 {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    log::debug!("now_unix: {}", ts);
    ts
}

fn load_prompts(path: &str) -> Result<Vec<PromptRow>> {
    log::info!("load_prompts: opening {}", path);
    let f = File::open(path)?;
    let br = BufReader::new(f);

    let mut rows = Vec::new();
    for (i, line) in br.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            log::debug!("load_prompts: skipping empty line {}", i);
            continue;
        }

        if i < 3 {
            log::debug!("load_prompts: line {} bytes={}", i, line.len());
        }

        let row = serde_json::from_str::<PromptRow>(&line)?;
        if i < 3 {
            log::debug!(
                "load_prompts: parsed row {} id={} tokens={}",
                i,
                row.id,
                row.tokens.len()
            );
        }
        rows.push(row);
    }

    log::info!("load_prompts: loaded {} prompt rows", rows.len());
    Ok(rows)
}

fn parse_usize_list(csv: &str) -> Result<Vec<usize>> {
    let s = csv.trim();
    log::debug!("parse_usize_list: raw='{}' trimmed='{}'", csv, s);

    if s.is_empty() {
        log::debug!("parse_usize_list: empty -> []");
        return Ok(vec![]);
    }

    let mut out = Vec::new();
    for (i, part) in s.split(',').enumerate() {
        let p = part.trim();
        if p.is_empty() {
            log::debug!("parse_usize_list: skipping empty segment {}", i);
            continue;
        }

        let v = p
            .parse::<usize>()
            .map_err(|e| anyhow!("bad int '{p}': {e}"))?;
        log::debug!("parse_usize_list: segment {} -> {}", i, v);
        out.push(v);
    }

    if out.is_empty() {
        return Err(anyhow!("empty list"));
    }
    log::info!("parse_usize_list: parsed {:?}", out);
    Ok(out)
}

fn percentile_ms(mut xs: Vec<f64>, p: f64) -> f64 {
    if xs.is_empty() {
        log::debug!("percentile_ms: empty input p={} -> 0.0", p);
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((xs.len() - 1) as f64 * p).round() as usize;
    let out = xs[idx];
    log::debug!(
        "percentile_ms: n={} p={} idx={} -> {:.6}",
        xs.len(),
        p,
        idx,
        out
    );
    out
}

fn current_rss_mb(sys: &mut System, pid: Pid) -> f64 {
    sys.refresh_processes(ProcessesToUpdate::All, true);
    let rss = sys
        .process(pid)
        .map(|p| p.memory() as f64 / 1024.0) // KiB -> MiB
        .unwrap_or(0.0);
    log::debug!("current_rss_mb: pid={:?} rss_mb={:.2}", pid, rss);
    rss
}

fn parse_tau_mode(s: &str) -> Result<ManifoldTauMode> {
    log::info!("parse_tau_mode: '{}'", s);
    match s.to_lowercase().as_str() {
        "median" => Ok(ManifoldTauMode::Median),
        "mean" => Ok(ManifoldTauMode::Mean),
        other => Err(anyhow!("unknown --tau-mode '{other}' (use median|mean)")),
    }
}

fn assert_logits_finite(logits: &Tensor<B, 3>) {
    let v: Vec<f32> = logits.clone().to_data().to_vec().unwrap();
    let ok = v.iter().all(|x| x.is_finite());
    log::debug!("assert_logits_finite: n={} finite={}", v.len(), ok);
    assert!(ok, "Found NaN/Inf in logits");
}

fn greedy_next_id(logits_step: Tensor<B, 3>) -> Tensor<B, 2, Int> {
    // logits_step: [B,1,V] -> next: [B,1]
    let [b, t, v] = logits_step.dims();
    log::debug!(
        "greedy_next_id: logits_step dims=[B,T,V]=[{}, {}, {}]",
        b,
        t,
        v
    );
    logits_step.reshape([b, v]).argmax(1).reshape([b, 1])
}

#[derive(Debug)]
struct BenchOut {
    prompt_len: usize,
    gen_tokens: usize,

    prefill_ms: f64,
    prime_ms: f64,

    ttft_ms: f64,
    decode_total_ms: f64,

    tokens_per_sec: f64,
    decode_tokens_per_sec: f64,

    end_to_end_ms: f64,

    p50_token_ms: f64,
    p95_token_ms: f64,

    rss_before_mb: f64,
    rss_after_mb: f64,

    per_token_ms: Vec<f64>,
}

fn materialize_prompt(base: &PromptRow, target_len: usize) -> PromptRow {
    if base.tokens.is_empty() {
        panic!("base token cannot be empty")
    }

    let base_len = base.tokens.len();
    let reps = (target_len + base_len - 1) / base_len; // ceil(target_len / base_len)
    log::debug!(
        "materialize_prompt: id={} base_len={} target_len={} reps={}",
        base.id,
        base_len,
        target_len,
        reps
    );

    let mut tokens = Vec::with_capacity(reps * base_len);
    for r in 0..reps {
        if r < 2 {
            log::debug!(
                "materialize_prompt: id={} extending rep {} ({} tokens)",
                base.id,
                r,
                base_len
            );
        }
        tokens.extend_from_slice(&base.tokens);
    }
    tokens.truncate(target_len);

    let out = PromptRow {
        id: format!("{}_L{}", base.id, target_len),
        tokens,
    };

    log::debug!(
        "materialize_prompt: out id={} len={}",
        out.id,
        out.tokens.len()
    );
    out
}

// ---------------- NanoGPT ----------------

fn bench_nanogpt(
    model: &NanoModel<B>,
    device: &<B as Backend>::Device,
    prompt: &PromptRow,
    mode: Mode,
    max_new_tokens: usize,
    sys: &mut System,
    pid: Pid,
) -> Result<BenchOut> {
    if prompt.tokens.is_empty() {
        return Err(anyhow!("prompt tokens must be non-empty"));
    }

    log::debug!(
        "bench_nanogpt: enter prompt_id={} prompt_len={} mode={:?} max_new_tokens={}",
        prompt.id,
        prompt.tokens.len(),
        mode,
        max_new_tokens
    );

    let rss_before = current_rss_mb(sys, pid);
    let prompt_len = prompt.tokens.len();

    log::debug!(
        "bench_nanogpt: building ids tensor from_ints len={}",
        prompt.tokens.len()
    );
    let ids =
        Tensor::<B, 1, Int>::from_ints(prompt.tokens.as_slice(), device).reshape([1, prompt_len]);
    log::debug!("bench_nanogpt: ids dims={:?}", ids.dims());

    match mode {
        Mode::NoCache => {
            let mut per_token_ms = Vec::with_capacity(max_new_tokens);
            let mut all_tokens = prompt.tokens.clone();

            // token 0 (TTFT)
            let t_first = Instant::now();
            let ctx = Tensor::<B, 1, Int>::from_ints(all_tokens.as_slice(), device)
                .reshape([1, all_tokens.len()]);
            log::debug!("bench_nanogpt NoCache: ctx len={}", all_tokens.len());

            let logits = model.forward(ctx, true);
            assert_logits_finite(&logits);
            let [b, t, v] = logits.dims();
            log::debug!("bench_nanogpt NoCache: logits dims=[{}, {}, {}]", b, t, v);

            let last = logits.slice([0..b, (t - 1)..t, 0..v]).reshape([b, v]);
            let next = last.argmax(1).reshape([1, 1]);
            let next_id: TokenId = next.to_data().as_slice::<TokenId>().unwrap()[0];
            all_tokens.push(next_id);

            let ttft_ms = t_first.elapsed().as_secs_f64() * 1000.0;
            per_token_ms.push(ttft_ms);

            // tokens 1..N-1
            let t_decode_all = Instant::now();
            for step in 1..max_new_tokens {
                if step <= 3 || step == max_new_tokens - 1 {
                    log::debug!(
                        "bench_nanogpt NoCache: step={} ctx_len={}",
                        step,
                        all_tokens.len()
                    );
                }
                let t_step = Instant::now();
                let ctx = Tensor::<B, 1, Int>::from_ints(all_tokens.as_slice(), device)
                    .reshape([1, all_tokens.len()]);
                let logits = model.forward(ctx, true);
                assert_logits_finite(&logits);
                let [b, t, v] = logits.dims();
                let last = logits.slice([0..b, (t - 1)..t, 0..v]).reshape([b, v]);
                let next = last.argmax(1).reshape([1, 1]);
                let next_id: TokenId = next.to_data().as_slice::<TokenId>().unwrap()[0];
                all_tokens.push(next_id);
                per_token_ms.push(t_step.elapsed().as_secs_f64() * 1000.0);
            }

            let decode_total_ms = t_decode_all.elapsed().as_secs_f64() * 1000.0;
            let decode_tokens = max_new_tokens.saturating_sub(1) as f64;
            let decode_tokens_per_sec = if decode_total_ms > 0.0 {
                decode_tokens / (decode_total_ms / 1000.0)
            } else {
                0.0
            };

            let rss_after = current_rss_mb(sys, pid);

            log::debug!(
                "bench_nanogpt NoCache: exit prompt_id={} ttft_ms={:.4} decode_total_ms={:.4}",
                prompt.id,
                ttft_ms,
                decode_total_ms
            );

            Ok(BenchOut {
                prompt_len,
                gen_tokens: max_new_tokens,
                prefill_ms: 0.0,
                prime_ms: 0.0,
                ttft_ms,
                decode_total_ms,
                tokens_per_sec: decode_tokens_per_sec,
                decode_tokens_per_sec,
                end_to_end_ms: ttft_ms + decode_total_ms,
                p50_token_ms: percentile_ms(per_token_ms.clone(), 0.50),
                p95_token_ms: percentile_ms(per_token_ms.clone(), 0.95),
                rss_before_mb: rss_before,
                rss_after_mb: rss_after,
                per_token_ms,
            })
        }

        Mode::KvCache => {
            let mut per_token_ms = Vec::with_capacity(max_new_tokens);

            // Prefill (cache-build) via decode-priming on prompt tokens except last.
            let mut cache = tauformer::engine::KVCache::<B>::new(model.num_layers());
            cache.clear();
            log::debug!(
                "bench_nanogpt KvCache: cache created+cleared num_layers={}",
                model.num_layers()
            );

            let t_prefill = Instant::now();
            if prompt_len > 1 {
                for pos in 0..(prompt_len - 1) {
                    if pos < 3 || pos + 1 == prompt_len - 1 {
                        log::debug!(
                            "bench_nanogpt KvCache: prefill pos={}/{}",
                            pos,
                            prompt_len - 1
                        );
                    }
                    let tok = ids.clone().slice([0..1, pos..pos + 1]); // [1,1]
                    let logits = model.forward_decode(tok, &mut cache, true);
                    assert_logits_finite(&logits);
                    cache.advance();
                }
            }
            let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
            log::debug!("bench_nanogpt KvCache: prefill_ms={:.4}", prefill_ms);

            // token 0 (TTFT): decode using last prompt token
            let mut last_id = ids.clone().slice([0..1, (prompt_len - 1)..prompt_len]); // [1,1]
            let t_first = Instant::now();
            let logits1 = model.forward_decode(last_id.clone(), &mut cache, true);
            assert_logits_finite(&logits1);
            let next = greedy_next_id(logits1);
            let ttft_ms = t_first.elapsed().as_secs_f64() * 1000.0;
            per_token_ms.push(ttft_ms);
            cache.advance();
            last_id = next;

            // tokens 1..N-1
            let t_decode = Instant::now();
            for step in 1..max_new_tokens {
                if step <= 3 || step == max_new_tokens - 1 {
                    log::debug!(
                        "bench_nanogpt KvCache: decode step={}/{}",
                        step,
                        max_new_tokens - 1
                    );
                }
                let t_step = Instant::now();
                let logits = model.forward_decode(last_id.clone(), &mut cache, true);
                assert_logits_finite(&logits);
                let next = greedy_next_id(logits);
                per_token_ms.push(t_step.elapsed().as_secs_f64() * 1000.0);
                cache.advance();
                last_id = next;
            }

            let decode_total_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
            let decode_tokens = max_new_tokens.saturating_sub(1) as f64;
            let decode_tokens_per_sec = if decode_total_ms > 0.0 {
                decode_tokens / (decode_total_ms / 1000.0)
            } else {
                0.0
            };

            let rss_after = current_rss_mb(sys, pid);
            let end_to_end_ms = prefill_ms + ttft_ms + decode_total_ms;

            log::debug!(
                "bench_nanogpt KvCache: exit prompt_id={} prefill_ms={:.4} ttft_ms={:.4} decode_total_ms={:.4}",
                prompt.id,
                prefill_ms,
                ttft_ms,
                decode_total_ms
            );

            Ok(BenchOut {
                prompt_len,
                gen_tokens: max_new_tokens,
                prefill_ms,
                prime_ms: 0.0,
                ttft_ms,
                decode_total_ms,
                tokens_per_sec: decode_tokens_per_sec,
                decode_tokens_per_sec,
                end_to_end_ms,
                p50_token_ms: percentile_ms(per_token_ms.clone(), 0.50),
                p95_token_ms: percentile_ms(per_token_ms.clone(), 0.95),
                rss_before_mb: rss_before,
                rss_after_mb: rss_after,
                per_token_ms,
            })
        }
    }
}

// ---------------- TauGPT ----------------

fn bench_taugpt(
    model: &TauGptModel<B>,
    device: &<B as Backend>::Device,
    prompt: &PromptRow,
    mode: Mode,
    max_new_tokens: usize,
    sys: &mut System,
    pid: Pid,
) -> Result<BenchOut> {
    if prompt.tokens.is_empty() {
        return Err(anyhow!("prompt tokens must be non-empty"));
    }

    log::debug!(
        "bench_taugpt: enter prompt_id={} prompt_len={} mode={:?} max_new_tokens={}",
        prompt.id,
        prompt.tokens.len(),
        mode,
        max_new_tokens
    );

    let rss_before = current_rss_mb(sys, pid);
    let prompt_len = prompt.tokens.len();

    let ids =
        Tensor::<B, 1, Int>::from_ints(prompt.tokens.as_slice(), device).reshape([1, prompt_len]);

    match mode {
        Mode::NoCache => {
            let mut per_token_ms = Vec::with_capacity(max_new_tokens);
            let mut all_tokens = prompt.tokens.clone();

            // token 0 (TTFT)
            let t_first = Instant::now();
            let ctx = Tensor::<B, 1, Int>::from_ints(all_tokens.as_slice(), device)
                .reshape([1, all_tokens.len()]);
            log::debug!("bench_taugpt NoCache: ctx len={}", all_tokens.len());

            let logits = model.forward(ctx, true);
            assert_logits_finite(&logits);
            let [b, t, v] = logits.dims();
            log::debug!("bench_taugpt NoCache: logits dims=[{}, {}, {}]", b, t, v);

            let last = logits.slice([0..b, (t - 1)..t, 0..v]).reshape([b, v]);
            let next = last.argmax(1).reshape([1, 1]);
            let next_id: TokenId = next.to_data().as_slice::<TokenId>().unwrap()[0];
            all_tokens.push(next_id);

            let ttft_ms = t_first.elapsed().as_secs_f64() * 1000.0;
            per_token_ms.push(ttft_ms);

            // tokens 1..N-1
            let t_decode_all = Instant::now();
            for step in 1..max_new_tokens {
                if step <= 3 || step == max_new_tokens - 1 {
                    log::debug!(
                        "bench_taugpt NoCache: step={} ctx_len={}",
                        step,
                        all_tokens.len()
                    );
                }
                let t_step = Instant::now();
                let ctx = Tensor::<B, 1, Int>::from_ints(all_tokens.as_slice(), device)
                    .reshape([1, all_tokens.len()]);
                let logits = model.forward(ctx, true);
                assert_logits_finite(&logits);
                let [b, t, v] = logits.dims();
                let last = logits.slice([0..b, (t - 1)..t, 0..v]).reshape([b, v]);
                let next = last.argmax(1).reshape([1, 1]);
                let next_id: TokenId = next.to_data().as_slice::<TokenId>().unwrap()[0];
                all_tokens.push(next_id);
                per_token_ms.push(t_step.elapsed().as_secs_f64() * 1000.0);
            }

            let decode_total_ms = t_decode_all.elapsed().as_secs_f64() * 1000.0;
            let decode_tokens = max_new_tokens.saturating_sub(1) as f64;
            let decode_tokens_per_sec = if decode_total_ms > 0.0 {
                decode_tokens / (decode_total_ms / 1000.0)
            } else {
                0.0
            };

            let rss_after = current_rss_mb(sys, pid);

            log::debug!(
                "bench_taugpt NoCache: exit prompt_id={} ttft_ms={:.4} decode_total_ms={:.4}",
                prompt.id,
                ttft_ms,
                decode_total_ms
            );

            Ok(BenchOut {
                prompt_len,
                gen_tokens: max_new_tokens,
                prefill_ms: 0.0,
                prime_ms: 0.0,
                ttft_ms,
                decode_total_ms,
                tokens_per_sec: decode_tokens_per_sec,
                decode_tokens_per_sec,
                end_to_end_ms: ttft_ms + decode_total_ms,
                p50_token_ms: percentile_ms(per_token_ms.clone(), 0.50),
                p95_token_ms: percentile_ms(per_token_ms.clone(), 0.95),
                rss_before_mb: rss_before,
                rss_after_mb: rss_after,
                per_token_ms,
            })
        }

        Mode::KvCache => {
            let mut per_token_ms = Vec::with_capacity(max_new_tokens);

            // Prefill (cache-build) via decode-priming
            let mut cache = TauKVCache::<B>::new(model.num_layers());
            cache.reset();
            log::debug!(
                "bench_taugpt KvCache: cache created+reset num_layers={}",
                model.num_layers()
            );

            let t_prefill = Instant::now();
            if prompt_len > 1 {
                for pos in 0..(prompt_len - 1) {
                    if pos < 3 || pos + 1 == prompt_len - 1 {
                        log::debug!(
                            "bench_taugpt KvCache: prefill pos={}/{}",
                            pos,
                            prompt_len - 1
                        );
                    }
                    let tok = ids.clone().slice([0..1, pos..pos + 1]); // [1,1]
                    let logits = model.forward_decode(tok, &mut cache, true);
                    assert_logits_finite(&logits);
                    cache.advance();
                }
            }
            let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
            log::debug!("bench_taugpt KvCache: prefill_ms={:.4}", prefill_ms);

            // token 0 (TTFT)
            let mut last_id = ids.clone().slice([0..1, (prompt_len - 1)..prompt_len]); // [1,1]
            let t_first = Instant::now();
            let logits1 = model.forward_decode(last_id.clone(), &mut cache, true);
            assert_logits_finite(&logits1);
            let next = greedy_next_id(logits1);
            let ttft_ms = t_first.elapsed().as_secs_f64() * 1000.0;
            per_token_ms.push(ttft_ms);
            cache.advance();
            last_id = next;

            // tokens 1..N-1
            let t_decode = Instant::now();
            for step in 1..max_new_tokens {
                if step <= 3 || step == max_new_tokens - 1 {
                    log::debug!(
                        "bench_taugpt KvCache: decode step={}/{}",
                        step,
                        max_new_tokens - 1
                    );
                }
                let t_step = Instant::now();
                let logits = model.forward_decode(last_id.clone(), &mut cache, true);
                assert_logits_finite(&logits);
                let next = greedy_next_id(logits);
                per_token_ms.push(t_step.elapsed().as_secs_f64() * 1000.0);
                cache.advance();
                last_id = next;
            }

            let decode_total_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
            let decode_tokens = max_new_tokens.saturating_sub(1) as f64;
            let decode_tokens_per_sec = if decode_total_ms > 0.0 {
                decode_tokens / (decode_total_ms / 1000.0)
            } else {
                0.0
            };

            let rss_after = current_rss_mb(sys, pid);
            let end_to_end_ms = prefill_ms + ttft_ms + decode_total_ms;

            log::debug!(
                "bench_taugpt KvCache: exit prompt_id={} prefill_ms={:.4} ttft_ms={:.4} decode_total_ms={:.4}",
                prompt.id,
                prefill_ms,
                ttft_ms,
                decode_total_ms
            );

            Ok(BenchOut {
                prompt_len,
                gen_tokens: max_new_tokens,
                prefill_ms,
                prime_ms: 0.0,
                ttft_ms,
                decode_total_ms,
                tokens_per_sec: decode_tokens_per_sec,
                decode_tokens_per_sec,
                end_to_end_ms,
                p50_token_ms: percentile_ms(per_token_ms.clone(), 0.50),
                p95_token_ms: percentile_ms(per_token_ms.clone(), 0.95),
                rss_before_mb: rss_before,
                rss_after_mb: rss_after,
                per_token_ms,
            })
        }
    }
}

fn write_headers(runs_w: &mut Writer<File>, tok_w: &mut Writer<File>) -> Result<()> {
    log::debug!("write_headers: writing CSV headers");
    runs_w.write_record([
        "run_id",
        "engine",
        "mode",
        "prompt_id",
        "prompt_len",
        "gen_tokens",
        "prefill_ms",
        "prime_ms",
        "ttft_ms",
        "decode_total_ms",
        "tokens_per_sec",
        "decode_tokens_per_sec",
        "end_to_end_ms",
        "p50_token_ms",
        "p95_token_ms",
        "rss_before_mb",
        "rss_after_mb",
    ])?;

    tok_w.write_record([
        "run_id",
        "engine",
        "mode",
        "prompt_id",
        "token_index",
        "token_ms",
    ])?;
    Ok(())
}

fn main() -> Result<()> {
    tauformer::init();
    let args = Args::parse();
    log::info!("Args: {:?}", args);

    print_backend_info();

    // Prompts
    let mut prompts = load_prompts(&args.prompts)?;
    if prompts.is_empty() {
        return Err(anyhow!("no prompts loaded"));
    }
    log::info!("main: loaded {} prompts", prompts.len());

    if args.max_prompts > 0 && prompts.len() > args.max_prompts {
        log::info!(
            "main: truncating prompts {} -> {}",
            prompts.len(),
            args.max_prompts
        );
        prompts.truncate(args.max_prompts);
    }
    log::info!("main: using {} prompts", prompts.len());

    // Sweep space
    let prompt_lens = {
        let parsed = parse_usize_list(&args.prompt_lens)?;
        if parsed.is_empty() {
            let default_lens = vec![384 * 2, 384 * 3];
            log::info!(
                "main: --prompt-lens empty, defaulting to {:?}",
                default_lens
            );
            default_lens
        } else {
            log::info!("main: using --prompt-lens {:?}", parsed);
            parsed
        }
    };

    let gen_tokens = {
        let gt = parse_usize_list(&args.gen_tokens)?;
        log::info!("main: gen_tokens sweep = {:?}", gt);
        gt
    };

    // Manifold/config
    let manifold_path = PathBuf::from(&args.manifold);
    log::info!("main: loading manifold from {}", manifold_path.display());
    let tau_mode = parse_tau_mode(&args.tau_mode)?;
    let manifold = load_domain_manifold(&manifold_path)?;
    let manifold_dim = manifold.nfeatures;
    log::info!("main: manifold_dim (nfeatures) = {}", manifold_dim);

    if args.n_head == 0 || args.n_kv_head == 0 {
        return Err(anyhow!("n_head and n_kv_head must be >= 1"));
    }
    if args.n_kv_head > args.n_head || (args.n_head % args.n_kv_head != 0) {
        return Err(anyhow!(
            "invalid MQA config: require n_kv_head <= n_head and n_head % n_kv_head == 0"
        ));
    }

    let max_prompt_len = *prompt_lens.iter().max().unwrap();
    let max_new_tokens = *gen_tokens.iter().max().unwrap();
    let seq_len = max_prompt_len + max_new_tokens + 2;
    log::info!(
        "main: max_prompt_len={} max_new_tokens={} -> seq_len={}",
        max_prompt_len,
        max_new_tokens,
        seq_len
    );

    let max_token_id = prompts
        .iter()
        .flat_map(|p| p.tokens.iter().copied())
        .max()
        .unwrap_or(0);

    let vocab_size = (max_token_id as usize + 1).max(64);
    log::info!(
        "main: max_token_id={} -> vocab_size={}",
        max_token_id,
        vocab_size
    );

    let cfg = NanoChatConfig {
        sequence_len: seq_len,
        vocab_size,
        n_layer: args.n_layer,
        n_head: args.n_head,
        n_kv_head: args.n_kv_head,
        n_embd: args.n_head * manifold_dim,
        block_size: seq_len,
        dropout: 0.0,
    };
    log::info!(
        "main: cfg sequence_len={} vocab_size={} n_layer={} n_head={} n_kv_head={} n_embd={}",
        cfg.sequence_len,
        cfg.vocab_size,
        cfg.n_layer,
        cfg.n_head,
        cfg.n_kv_head,
        cfg.n_embd
    );

    let device = get_device();
    log::info!("main: got device");

    // Output dir: output/<timestamp>/{runs.csv, token_latencies.csv}
    let variant = now_unix().to_string();
    let out_dir = Path::new(&args.out_dir).join(&variant);
    log::info!("main: creating out_dir {}", out_dir.display());
    create_dir_all(&out_dir)?;

    let runs_path = out_dir.join("runs.csv");
    let tok_path = out_dir.join("token_latencies.csv");
    log::info!(
        "main: output files runs={} tokens={}",
        runs_path.display(),
        tok_path.display()
    );

    let mut runs_w = Writer::from_path(&runs_path)?;
    let mut tok_w = Writer::from_path(&tok_path)?;
    write_headers(&mut runs_w, &mut tok_w)?;

    let mut sys = System::new_all();
    let pid = Pid::from(std::process::id() as usize);
    log::info!("main: pid={:?}", pid);

    // Build models once (cfg supports max sweep)
    let nano = if args.run_nano {
        log::info!("main: building NanoModel...");
        let m = NanoModel::<B>::new(&cfg, &device);
        log::info!("main: NanoModel built");
        Some(m)
    } else {
        log::info!("main: NanoModel disabled");
        None
    };

    let tau = if args.run_tau {
        log::info!("main: building TauGptModel (sparse laplacian)...");
        let m =
            TauGptModel::<B>::new_with_sparse_laplacian(&cfg, &manifold_path, &device, tau_mode);
        log::info!("main: TauGptModel built");
        Some(m)
    } else {
        log::info!("main: TauGptModel disabled");
        None
    };

    let modes = [Mode::NoCache, Mode::KvCache];
    let mut run_id: u64 = 0;

    for &pl in &prompt_lens {
        log::info!("main: ===== prompt_len={} =====", pl);

        // Materialize prompt variants at this prompt_len
        let prompts_pl: Vec<PromptRow> =
            prompts.iter().map(|p| materialize_prompt(p, pl)).collect();
        log::info!(
            "main: materialized {} prompts at len={}",
            prompts_pl.len(),
            pl
        );

        for &gt in &gen_tokens {
            log::info!("main: ---- gen_tokens={} ----", gt);

            for p in &prompts_pl {
                log::info!("main: prompt_id={} prompt_len={}", p.id, p.tokens.len());

                for &m in &modes {
                    let mode_str = match m {
                        Mode::NoCache => "no_cache",
                        Mode::KvCache => "kv_cache",
                    };
                    log::info!("main: mode={}", mode_str);

                    if let Some(nano) = nano.as_ref() {
                        run_id += 1;
                        log::info!("main: run_id={} engine=nano", run_id);

                        let out = bench_nanogpt(nano, &device, p, m, gt, &mut sys, pid)?;
                        runs_w.write_record([
                            run_id.to_string(),
                            "nano".to_string(),
                            mode_str.to_string(),
                            p.id.clone(),
                            out.prompt_len.to_string(),
                            out.gen_tokens.to_string(),
                            format!("{:.4}", out.prefill_ms),
                            format!("{:.4}", out.prime_ms),
                            format!("{:.4}", out.ttft_ms),
                            format!("{:.4}", out.decode_total_ms),
                            format!("{:.4}", out.tokens_per_sec),
                            format!("{:.4}", out.decode_tokens_per_sec),
                            format!("{:.4}", out.end_to_end_ms),
                            format!("{:.4}", out.p50_token_ms),
                            format!("{:.4}", out.p95_token_ms),
                            format!("{:.4}", out.rss_before_mb),
                            format!("{:.4}", out.rss_after_mb),
                        ])?;

                        for (i, ms) in out.per_token_ms.iter().enumerate() {
                            tok_w.write_record([
                                run_id.to_string(),
                                "nano".to_string(),
                                mode_str.to_string(),
                                p.id.clone(),
                                i.to_string(),
                                format!("{:.6}", ms),
                            ])?;
                        }
                    }

                    if let Some(tau) = tau.as_ref() {
                        run_id += 1;
                        log::info!("main: run_id={} engine=tau", run_id);

                        let out = bench_taugpt(tau, &device, p, m, gt, &mut sys, pid)?;
                        runs_w.write_record([
                            run_id.to_string(),
                            "tau".to_string(),
                            mode_str.to_string(),
                            p.id.clone(),
                            out.prompt_len.to_string(),
                            out.gen_tokens.to_string(),
                            format!("{:.4}", out.prefill_ms),
                            format!("{:.4}", out.prime_ms),
                            format!("{:.4}", out.ttft_ms),
                            format!("{:.4}", out.decode_total_ms),
                            format!("{:.4}", out.tokens_per_sec),
                            format!("{:.4}", out.decode_tokens_per_sec),
                            format!("{:.4}", out.end_to_end_ms),
                            format!("{:.4}", out.p50_token_ms),
                            format!("{:.4}", out.p95_token_ms),
                            format!("{:.4}", out.rss_before_mb),
                            format!("{:.4}", out.rss_after_mb),
                        ])?;

                        for (i, ms) in out.per_token_ms.iter().enumerate() {
                            tok_w.write_record([
                                run_id.to_string(),
                                "tau".to_string(),
                                mode_str.to_string(),
                                p.id.clone(),
                                i.to_string(),
                                format!("{:.6}", ms),
                            ])?;
                        }
                    }

                    runs_w.flush()?;
                    tok_w.flush()?;
                    log::debug!("main: flushed CSV writers (run_id={})", run_id);
                }
            }
        }
    }

    eprintln!(
        "Wrote:\n  {}\n  {}",
        runs_path.display(),
        tok_path.display()
    );
    Ok(())
}
