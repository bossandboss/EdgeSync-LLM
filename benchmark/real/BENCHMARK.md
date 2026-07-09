# EdgeSync-LLM — Measured TTFT Benchmark (`benchmark/real/`)

This is a **falsifiable, on-device** benchmark. Unlike `benchmark/runner.go`, it
contains **no cost formulas and no hardware constants**. Every latency is a real
`time.Now()` delta around a real adapter call (`Generate`, `InjectFragment`,
`ExtractFragment`). The headline metric is **time-to-first-token (TTFT)** — the
prefill-dominated latency that KV-fragment reuse is designed to remove.

## Why the old benchmark could not validate anything

`benchmark/runner.go` computes latency as `tokens × MsPerTokenPrefill`. Its output
is a deterministic function of its input constants, so its `validateRange` checks
test the model against itself and always pass. That is exactly why the README
numbers are labelled *"Expected results"*. This harness replaces the model with a
measurement.

## Two build modes

**Host smoke test (default, SIMULATED).** No model, no CGO adapter. A mock engine
exercises the full control flow so you can verify the harness compiles and runs.
Every output is stamped `SIMULATED` and the JSON manifest sets `"measurement": false`.
These numbers are **not** a device result and must never be quoted.

```bash
go run ./benchmark/real/ -n 400
```

**Real device (MEASURED).** Requires CGO + llama.cpp (`llama.h`/`libllama`) per the
README build section, and a `.gguf` model. This wires `adapter.LoadLlamaCppModel`,
so every latency is a real prefill/inject/decode. Output is stamped `MEASURED` and
`"measurement": true`.

```bash
# on the Android target (or a desktop with the C deps linked)
go build -tags realdevice -o edgebench ./benchmark/real/
./edgebench \
  -model-path /data/local/tmp/qwen2.5-0.5b-q4_k_m.gguf \
  -layers 24 -kv-heads 8 -head-dim 64 -nctx 4096 \
  -embed-dir /data/local/tmp/models \
  -n 500 -repeats 5 -prefix-share 0.7
```

Set `-layers/-kv-heads/-head-dim/-quant/-arch` to match the GGUF you load —
fragments are rejected on `ModelID` mismatch.

## What it measures

- **cold TTFT** — `ClearKVCache` then generate 1 token (full prefill).
- **fragment TTFT (all)** — end-to-end including misses; the honest deployment number.
- **fragment TTFT (hits)** — hits only; the best case.
- **speedup** with a significance flag: `true` only when the 95% CIs on the means
  are disjoint, so you cannot report a speedup that is inside the noise floor.
- **first-token match rate** — a KV cache that changes outputs is a bug, not a
  speedup. Hits whose first token diverges from the cold path trigger a
  `CORRECTNESS WARNING`. **Do not publish a latency number if this fires.**
- **fragment memory** — real bytes from `KVFragment.SizeBytes()`, plus process
  `VmRSS` delta.

## Reproducibility

Each run writes `results/bench-<tag>-<timestamp>.json` (full manifest: git commit,
CPU model, GOOS/GOARCH, engine + model + embedder identity, seed, corpus hash,
every config value) and `.csv` (per-request rows). Same seed + same binary + same
device ⇒ same corpus (`corpus_hash` confirms it).

## Turning this into a citable README number

1. Run on the real device with the real MiniLM `.ort` present (so hit rate reflects
   real semantics, not the hash fallback).
2. Confirm `first-token match ≈ 100%` (no correctness warning).
3. Confirm `speedup significant = true`.
4. Replace the README "Expected results" block with the measured `mean` / `p95`
   TTFT and speedup, and commit the `.json` manifest alongside as evidence.
