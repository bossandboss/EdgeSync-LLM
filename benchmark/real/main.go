package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// Manifest is the full, self-describing record of one run. Written to disk so
// any result is reproducible and auditable — this is what lets a README cite
// "measured", with the exact conditions attached.
type Manifest struct {
	Schema      string  `json:"schema"`
	Measurement bool    `json:"measurement"` // false ⇒ SIMULATED (mock engine)
	Timestamp   string  `json:"timestamp_utc"`
	GitCommit   string  `json:"git_commit"`
	Engine      string  `json:"engine"`
	EngineVer   string  `json:"engine_version"`
	Model       string  `json:"model"`
	Embedder    string  `json:"embedder"`
	GOARCH      string  `json:"goarch"`
	GOOS        string  `json:"goos"`
	NumCPU      int     `json:"num_cpu"`
	CPUModel    string  `json:"cpu_model"`
	Config      Config  `json:"config"`
	CorpusHash  string  `json:"corpus_hash"`
	DurationSec float64 `json:"duration_sec"`
	Result      *Report `json:"result"`
}

func main() {
	cfg := Config{}
	flag.IntVar(&cfg.N, "n", 500, "number of requests")
	flag.IntVar(&cfg.MaxGen, "maxgen", 1, "tokens to generate per request (TTFT uses 1)")
	flag.IntVar(&cfg.Repeats, "repeats", 3, "timed repeats per call (median taken)")
	flag.IntVar(&cfg.Warmup, "warmup", 1, "warmup calls discarded before timing")
	flag.Int64Var(&cfg.Seed, "seed", 42, "corpus RNG seed")
	flag.Float64Var(&cfg.PrefixShare, "prefix-share", 0.7, "fraction of requests reusing a shared prefix")
	flag.StringVar(&cfg.EmbedDir, "embed-dir", "./models", "dir containing MiniLM .ort (falls back to hash encoder if absent)")

	// Adapter (used by -tags realdevice build; ignored by mock).
	flag.StringVar(&cfg.Adapter.ModelPath, "model-path", "", "path to .gguf (realdevice build)")
	flag.StringVar(&cfg.Adapter.Arch, "arch", "qwen", "model architecture")
	flag.StringVar(&cfg.Adapter.ModelName, "model-name", "Qwen2.5-0.5B", "model name")
	flag.StringVar(&cfg.Adapter.Quant, "quant", "Q4_K_M", "quantization")
	flag.IntVar(&cfg.Adapter.NCtx, "nctx", 4096, "context length")
	flag.IntVar(&cfg.Adapter.NThreads, "threads", runtime.NumCPU(), "inference threads")
	flag.IntVar(&cfg.Adapter.NGpuLayers, "gpu-layers", 0, "GPU offload layers")
	flag.IntVar(&cfg.Adapter.HeadDim, "head-dim", 64, "KV head dim")
	flag.IntVar(&cfg.Adapter.NumKVHeads, "kv-heads", 8, "num KV heads")
	flag.IntVar(&cfg.Adapter.NumLayers, "layers", 24, "num transformer layers")

	flag.BoolVar(&cfg.Strict, "strict", true, "abort on the first engine error instead of timing the failure path")
	outDir := flag.String("out", "./results", "output directory")
	flag.Parse()

	reqs := buildCorpus(cfg.N, cfg.PrefixShare, cfg.Seed)

	h, err := newHarness(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, "init error:", err)
		os.Exit(1)
	}

	printBanner()

	// PREFLIGHT: prove the engine really decodes before collecting any numbers.
	ptoks, tokps, perr := h.Preflight(context.Background(), reqs[0])
	if perr != nil {
		fmt.Fprintln(os.Stderr, "PREFLIGHT FAILED:", perr)
		fmt.Fprintln(os.Stderr, "No numbers are reported. Fix the engine, then re-run.")
		os.Exit(1)
	}
	fmt.Printf("preflight: prompt=%d tokens, prefill=%.0f tok/s (%.1f ms)\n\n",
		ptoks, tokps, float64(ptoks)/tokps*1000)
	if tokps > 8000 {
		fmt.Fprintln(os.Stderr, "WARNING: implausibly high prefill throughput. The engine may not be")
		fmt.Fprintln(os.Stderr, "decoding. Treat every latency below as suspect until explained.")
	}

	start := time.Now()
	rep, runErr := h.Run(reqs)
	dur := time.Since(start).Seconds()
	if runErr != nil {
		fmt.Fprintln(os.Stderr, "RUN ABORTED:", runErr)
		fmt.Fprintln(os.Stderr, "No numbers are reported (use -strict=false to continue past errors).")
		os.Exit(1)
	}
	rep.PromptTokens = ptoks
	rep.PrefillTokPS = tokps

	man := &Manifest{
		Schema:      "edgesync.bench/v1",
		Measurement: engineReal,
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
		GitCommit:   gitCommit(),
		Engine:      h.ad.EngineName(),
		EngineVer:   h.ad.EngineVersion(),
		Model:       h.ad.ModelID().String(),
		Embedder:    encoderName(h),
		GOARCH:      runtime.GOARCH,
		GOOS:        runtime.GOOS,
		NumCPU:      runtime.NumCPU(),
		CPUModel:    cpuModel(),
		Config:      cfg,
		CorpusHash:  corpusHash(reqs),
		DurationSec: dur,
		Result:      rep,
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir:", err)
		os.Exit(1)
	}
	stamp := time.Now().UTC().Format("20060102-150405")
	tag := "measured"
	if !engineReal {
		tag = "SIMULATED"
	}
	base := filepath.Join(*outDir, fmt.Sprintf("bench-%s-%s", tag, stamp))
	writeJSON(base+".json", man)
	writeCSV(base+".csv", rep.Samples)

	printTable(man)
	fmt.Printf("\nManifest: %s.json\nPer-request CSV: %s.csv\n", base, base)
}

func printBanner() {
	if engineReal {
		return
	}
	line := strings.Repeat("!", 78)
	fmt.Println(line)
	fmt.Println("!!  SIMULATED RUN — mock engine, NO real model loaded.")
	fmt.Println("!!  These numbers exercise the harness only. They are NOT a device")
	fmt.Println("!!  measurement and MUST NOT be quoted as performance results.")
	fmt.Println("!!  Build with:  go build -tags realdevice ./benchmark/real/  on the")
	fmt.Println("!!  target device with a real .gguf to produce measured numbers.")
	fmt.Println(line)
}

func printTable(m *Manifest) {
	r := m.Result
	fmt.Println()
	fmt.Printf("EdgeSync-LLM TTFT benchmark  [%s]\n", statusWord(m.Measurement))
	fmt.Printf("  engine=%s  model=%s  embedder=%s\n", m.Engine, m.Model, m.Embedder)
	fmt.Printf("  arch=%s/%s  cpu=%q  cores=%d  commit=%s\n", m.GOOS, m.GOARCH, m.CPUModel, m.NumCPU, m.GitCommit)
	fmt.Printf("  requests=%d  repeats=%d  prefix-share=%.2f  seed=%d  corpus=%s\n",
		m.Config.N, m.Config.Repeats, m.Config.PrefixShare, m.Config.Seed, m.CorpusHash)
	fmt.Println("  ─────────────────────────────────────────────────────────────")
	fmt.Printf("  %-16s %9s %9s %9s %9s\n", "TTFT (ms)", "mean", "p50", "p95", "p99")
	row := func(name string, d Dist) {
		fmt.Printf("  %-16s %9.2f %9.2f %9.2f %9.2f\n", name, d.Mean, d.P50, d.P95, d.P99)
	}
	row("cold (no cache)", r.ColdTTFT)
	row("fragment all", r.FragTTFTAll)
	row("fragment hits", r.FragTTFTHits)
	fmt.Println("  ─────────────────────────────────────────────────────────────")
	fmt.Printf("  hit rate           %.1f%%  (exact=%d partial=%d miss=%d)\n",
		r.HitRatePct, r.ExactHits, r.PartialHits, r.Misses)
	fmt.Printf("  speedup (all)      %.2fx   significant=%v (95%% CIs disjoint)\n", r.SpeedupAll, r.SigDisjoint)
	fmt.Printf("  speedup (hits)     %.2fx\n", r.SpeedupHits)
	fmt.Printf("  cold TTFT 95%% CI   [%.2f, %.2f] ms\n", r.ColdTTFT.CI95Lo, r.ColdTTFT.CI95Hi)
	fmt.Printf("  frag TTFT 95%% CI   [%.2f, %.2f] ms\n", r.FragTTFTAll.CI95Lo, r.FragTTFTAll.CI95Hi)
	fmt.Printf("  first-token match  %.1f%%  (%d/%d hits identical to cold)\n",
		r.MatchRatePct, r.CorrectHits, r.TotalHits)
	fmt.Printf("  fragments stored   %d   total=%.2f MB   avg=%.2f MB\n",
		r.FragmentCount, r.FragTotalMB, r.FragAvgMB)
	fmt.Printf("  process RSS delta  %.1f MB\n", r.RSSDeltaMB)
	fmt.Printf("  engine errors      generate=%d extract=%d inject=%d\n",
		r.GenErrors, r.ExtractErrors, r.InjectErrors)
	if r.GenErrors+r.ExtractErrors+r.InjectErrors > 0 {
		fmt.Println("\n  ⚠ ENGINE ERRORS OCCURRED. Latencies above may include failure paths.")
	}

	if r.TotalHits > 0 && r.MatchRatePct < 99.0 {
		fmt.Println("\n  ⚠ CORRECTNESS WARNING: injected-fragment output diverges from cold")
		fmt.Println("    output on some hits. A KV cache that changes results is a bug, not")
		fmt.Println("    a speedup. Investigate before publishing any latency number.")
	}
}

func statusWord(measured bool) string {
	if measured {
		return "MEASURED"
	}
	return "SIMULATED — not a device measurement"
}

func writeJSON(path string, v any) {
	f, err := os.Create(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "write json:", err)
		return
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func writeCSV(path string, samples []Sample) {
	f, err := os.Create(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "write csv:", err)
		return
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	_ = w.Write([]string{"index", "cluster", "status", "ttft_ms", "inject_ms", "frag_bytes"})
	for _, s := range samples {
		_ = w.Write([]string{
			strconv.Itoa(s.Index),
			strconv.Itoa(s.ClusterID),
			s.Status,
			strconv.FormatFloat(s.TTFTms, 'f', 3, 64),
			strconv.FormatFloat(s.InjectMs, 'f', 3, 64),
			strconv.Itoa(s.FragBytes),
		})
	}
}

func gitCommit() string {
	out, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

func cpuModel() string {
	b, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return "unknown"
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "model name") || strings.HasPrefix(line, "Hardware") || strings.HasPrefix(line, "Processor") {
			if i := strings.Index(line, ":"); i >= 0 {
				return strings.TrimSpace(line[i+1:])
			}
		}
	}
	return "unknown"
}

func encoderName(h *harness) string {
	type named interface{ Name() string }
	if n, ok := h.enc.(named); ok {
		return n.Name()
	}
	return "unknown"
}
