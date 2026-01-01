use anyhow::{Result, anyhow};
use clap::Parser;
use csv::Writer;
use serde::Deserialize;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
    time::Instant,
};
use sysinfo::{Pid, ProcessesToUpdate, System};

use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};

use tauformer::{
    backend::{AutoBackend, get_device, print_backend_info},
    causalattention::GptModel as NanoModel,
    config::NanoChatConfig,
    engine::GptCache,
    pretraining::parquet::{TauMode as ManifoldTauMode, load_domain_manifold},
    taugpt::{TauGptModel, TauKVCache},
};

type B = AutoBackend;

#[derive(Parser, Debug)]
struct Args {
    /// JSONL file: {"id": "...", "tokens": [1,2,3]}
    #[arg(long)]
    prompts: String,

    /// Number of tokens to generate (per run)
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    /// Repeats per prompt per mode
    #[arg(long, default_value_t = 5)]
    repeats: usize,

    /// Warmup runs per prompt per mode (not written to CSV)
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Output CSV: per-run aggregates
    #[arg(long, default_value = "runs.csv")]
    runs_csv: String,

    /// Output CSV: per-token step latencies
    #[arg(long, default_value = "token_latencies.csv")]
    token_csv: String,

    /// Run TauGPT too (in addition to NanoGPT)
    #[arg(long, default_value_t = false)]
    run_tau: bool,

    /// Run TauGPT too (in addition to NanoGPT)
    #[arg(long, default_value_t = false)]
    run_nano: bool,

    /// Path to the manifold parquet used by TauGPT sparse constructor (same as tests).
    #[arg(long, default_value = "./domain_manifold/manifold.parquet")]
    manifold: String,

    /// Tau mode ("median" | "mean")
    #[arg(long, default_value = "median")]
    tau_mode: String,

    /// Heads for NanoGPT config (defaults to single-head benchmark mode).
    /// For manifold-native multihead, keep n_embd = n_head * manifold_dim.
    #[arg(long, default_value_t = 1)]
    n_head: usize,

    /// KV heads for NanoGPT config.
    #[arg(long, default_value_t = 1)]
    n_kv_head: usize,

    /// Layers for both models
    #[arg(long, default_value_t = 6)]
    n_layer: usize,
}

#[cfg(any(feature = "wgpu", feature = "cuda"))]
type TokenId = i32;
#[cfg(not(any(feature = "wgpu", feature = "cuda")))]
type TokenId = i64;

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

fn load_prompts(path: &str) -> Result<Vec<PromptRow>> {
    log::info!("Loading prompts from {}", path);
    let f = File::open(path)?;
    let br = BufReader::new(f);
    let mut rows = Vec::new();

    for (i, line) in br.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if i < 3 {
            log::debug!("Prompt JSONL line {}: {} bytes", i, line.len());
        }
        rows.push(serde_json::from_str::<PromptRow>(&line)?);
    }

    log::info!("Loaded {} prompt rows", rows.len());
    Ok(rows)
}

fn percentile_ms(mut xs: Vec<f64>, p: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((xs.len() - 1) as f64 * p).round() as usize;
    xs[idx]
}

fn current_rss_mb(sys: &mut System, pid: Pid) -> f64 {
    sys.refresh_processes(ProcessesToUpdate::All, true);
    sys.process(pid)
        .map(|p| p.memory() as f64 / 1024.0) // KiB -> MiB
        .unwrap_or(0.0)
}

fn parse_tau_mode(s: &str) -> Result<ManifoldTauMode> {
    log::info!("Parsing tau mode '{}'", s);
    match s.to_lowercase().as_str() {
        "median" => Ok(ManifoldTauMode::Median),
        "mean" => Ok(ManifoldTauMode::Mean),
        other => Err(anyhow!("unknown --tau-mode '{other}' (use median|mean)")),
    }
}

fn assert_logits_finite(logits: &Tensor<B, 3>) {
    let v: Vec<f32> = logits.clone().to_data().to_vec().unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "Found NaN/Inf in logits");
}

fn greedy_next_id(logits_step: Tensor<B, 3>) -> Tensor<B, 2, Int> {
    // logits_step: [B,1,V] -> next: [B,1]
    let [b, _t, v] = logits_step.dims();
    logits_step.reshape([b, v]).argmax(1).reshape([b, 1])
}

#[derive(Debug)]
struct BenchOut {
    prompt_len: usize,
    gen_tokens: usize,

    // TRAINING-LIKE prompt cost
    // - In no_cache mode this is 0 (because ttft_ms already covers it)
    // - In kv_cache mode this is the cache-build time (decode-priming loop)
    prefill_ms: f64,

    // kept for backward compatibility; always 0 now (we no longer measure "prime" separately)
    prime_ms: f64,

    // First token latency (token_index 0)
    ttft_ms: f64,

    // Remaining decode loop (token_index 1..gen_tokens-1)
    decode_total_ms: f64,

    // Old field kept, but now defined as decode_tokens_per_sec
    tokens_per_sec: f64,

    // New: correct decode TPS explicitly
    decode_tokens_per_sec: f64,

    // New: inference end-to-end time (cache build + TTFT + decode loop)
    end_to_end_ms: f64,

    p50_token_ms: f64,
    p95_token_ms: f64,
    rss_before_mb: f64,
    rss_after_mb: f64,
    per_token_ms: Vec<f64>,
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

    let rss_before = current_rss_mb(sys, pid);
    let prompt_len = prompt.tokens.len();
    let ids =
        Tensor::<B, 1, Int>::from_ints(prompt.tokens.as_slice(), device).reshape([1, prompt_len]);

    match mode {
        Mode::NoCache => {
            let mut per_token_ms = Vec::with_capacity(max_new_tokens);
            let mut all_tokens = prompt.tokens.clone();

            // token 0
            let t_first = Instant::now();
            {
                let ctx = Tensor::<B, 1, Int>::from_ints(all_tokens.as_slice(), device)
                    .reshape([1, all_tokens.len()]);
                let logits = model.forward(ctx, true);
                assert_logits_finite(&logits);
                let [b, t, v] = logits.dims();
                let last = logits.slice([0..b, (t - 1)..t, 0..v]).reshape([b, v]);
                let next = last.argmax(1).reshape([1, 1]);
                let next_id: TokenId = next.to_data().as_slice::<TokenId>().unwrap()[0];
                all_tokens.push(next_id);
            }
            let ttft_ms = t_first.elapsed().as_secs_f64() * 1000.0;
            per_token_ms.push(ttft_ms);

            // tokens 1..N-1
            let t_decode_all = Instant::now();
            for _step in 1..max_new_tokens {
                log::debug!("nanoGPT NoCache step {:?}", _step);
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

            Ok(BenchOut {
                prompt_len,
                gen_tokens: max_new_tokens,
                prefill_ms: 0.0,
                prime_ms: 0.0,
                ttft_ms,
                decode_total_ms,
                tokens_per_sec: decode_tokens_per_sec, // keep old column meaningful
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

            let t_prefill = Instant::now();
            if prompt_len > 1 {
                for pos in 0..(prompt_len - 1) {
                    let tok = ids.clone().slice([0..1, pos..pos + 1]); // [1,1]
                    let logits = model.forward_decode(tok, &mut cache, true);
                    assert_logits_finite(&logits);
                    cache.advance();
                }
            }
            let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

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
            for _step in 1..max_new_tokens {
                log::debug!("nanoGPT KvCache step {:?}", _step);
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

    let rss_before = current_rss_mb(sys, pid);
    let prompt_len = prompt.tokens.len();
    let ids =
        Tensor::<B, 1, Int>::from_ints(prompt.tokens.as_slice(), device).reshape([1, prompt_len]);

    match mode {
        Mode::NoCache => {
            let mut per_token_ms = Vec::with_capacity(max_new_tokens);
            let mut all_tokens = prompt.tokens.clone();

            // token 0
            let t_first = Instant::now();
            {
                let ctx = Tensor::<B, 1, Int>::from_ints(all_tokens.as_slice(), device)
                    .reshape([1, all_tokens.len()]);
                let logits = model.forward(ctx, true);
                assert_logits_finite(&logits);
                let [b, t, v] = logits.dims();
                let last = logits.slice([0..b, (t - 1)..t, 0..v]).reshape([b, v]);
                let next = last.argmax(1).reshape([1, 1]);
                let next_id: TokenId = next.to_data().as_slice::<TokenId>().unwrap()[0];
                all_tokens.push(next_id);
            }
            let ttft_ms = t_first.elapsed().as_secs_f64() * 1000.0;
            per_token_ms.push(ttft_ms);

            // tokens 1..N-1
            let t_decode_all = Instant::now();
            for _step in 1..max_new_tokens {
                log::debug!("tauGPT NoCache step {:?}", _step);
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

            let t_prefill = Instant::now();
            if prompt_len > 1 {
                for pos in 0..(prompt_len - 1) {
                    let tok = ids.clone().slice([0..1, pos..pos + 1]); // [1,1]
                    let logits = model.forward_decode(tok, &mut cache, true);
                    assert_logits_finite(&logits);
                    cache.advance(); // fixed symmetry with NanoKVCache
                }
            }
            let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

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
            for _step in 1..max_new_tokens {
                log::debug!("nanoGPT KvCache step {:?}", _step);
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

fn run_model(
    engine_label: &str,
    prompts: &[PromptRow],
    warmup: usize,
    repeats: usize,
    max_new_tokens: usize,
    sys: &mut System,
    pid: Pid,
    runs_w: &mut Writer<File>,
    tok_w: &mut Writer<File>,
    mut bench_one: impl FnMut(&PromptRow, Mode, &mut System, Pid) -> Result<BenchOut>,
    run_id: &mut u64,
) -> Result<()> {
    let modes = [Mode::NoCache, Mode::KvCache];

    for p in prompts {
        log::info!("Model {:?}", engine_label);
        log::info!("Starting prompt {:?}", p.id);
        for _ in 0..warmup {
            for &m in &modes {
                let _ = bench_one(p, m, sys, pid)?;
            }
        }

        for _rep in 0..repeats {
            log::info!("Repeat no. {:?}", _rep);
            for &m in &modes {
                *run_id += 1;

                let mode_str = match m {
                    Mode::NoCache => "no_cache",
                    Mode::KvCache => "kv_cache",
                };

                let out = bench_one(p, m, sys, pid)?;

                runs_w.write_record([
                    run_id.to_string(),
                    engine_label.to_string(),
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
                        engine_label.to_string(),
                        mode_str.to_string(),
                        p.id.clone(),
                        i.to_string(),
                        format!("{:.6}", ms),
                    ])?;
                }
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    tauformer::init();

    let args = Args::parse();
    log::info!("Args: {:?}", args);
    print_backend_info();

    let prompts = load_prompts(&args.prompts)?;
    if prompts.is_empty() {
        return Err(anyhow!("no prompts loaded"));
    }

    let manifold_path = PathBuf::from(&args.manifold);
    let tau_mode = parse_tau_mode(&args.tau_mode)?;

    // Load manifold metadata to get manifold_dim (nfeatures), as tests do.
    let manifold = load_domain_manifold(&manifold_path)?;
    let manifold_dim = manifold.nfeatures;

    if args.n_head == 0 || args.n_kv_head == 0 {
        return Err(anyhow!("n_head and n_kv_head must be >= 1"));
    }
    if args.n_kv_head > args.n_head || (args.n_head % args.n_kv_head != 0) {
        return Err(anyhow!(
            "invalid MQA config: require n_kv_head <= n_head and n_head % n_kv_head == 0"
        ));
    }

    let max_prompt_len = prompts.iter().map(|p| p.tokens.len()).max().unwrap_or(1);
    let seq_len = max_prompt_len + args.max_new_tokens + 2;

    let max_token_id = prompts
        .iter()
        .flat_map(|p| p.tokens.iter().copied())
        .max()
        .unwrap_or(0);
    let vocab_size = (max_token_id as usize + 1).max(64);

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

    let device = get_device();

    let mut sys = System::new_all();
    let pid = Pid::from(std::process::id() as usize);

    let mut runs_w = Writer::from_path(&args.runs_csv)?;
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

    let mut tok_w = Writer::from_path(&args.token_csv)?;
    tok_w.write_record([
        "run_id",
        "engine",
        "mode",
        "prompt_id",
        "token_index",
        "token_ms",
    ])?;

    let mut run_id: u64 = 0;

    if args.run_nano {
        // NanoGPT
        let nano = NanoModel::<B>::new(&cfg, &device);
        run_model(
            "nano",
            &prompts,
            args.warmup,
            args.repeats,
            args.max_new_tokens,
            &mut sys,
            pid,
            &mut runs_w,
            &mut tok_w,
            |p, m, sys, pid| bench_nanogpt(&nano, &device, p, m, args.max_new_tokens, sys, pid),
            &mut run_id,
        )?;
    }

    // TauGPT
    if args.run_tau {
        let tau =
            TauGptModel::<B>::new_with_sparse_laplacian(&cfg, &manifold_path, &device, tau_mode);
        run_model(
            "tau",
            &prompts,
            args.warmup,
            args.repeats,
            args.max_new_tokens,
            &mut sys,
            pid,
            &mut runs_w,
            &mut tok_w,
            |p, m, sys, pid| bench_taugpt(&tau, &device, p, m, args.max_new_tokens, sys, pid),
            &mut run_id,
        )?;
    }

    runs_w.flush()?;
    tok_w.flush()?;
    Ok(())
}
