package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/bossandboss/EdgeSync-LLM/cache"
	"github.com/bossandboss/EdgeSync-LLM/core"
	"github.com/bossandboss/EdgeSync-LLM/embedding"
)

// AdapterConfig is consumed by the build-tagged newAdapter() in either
// adapter_mock.go (host) or adapter_device.go (-tags realdevice).
type AdapterConfig struct {
	ModelPath  string
	Arch       string
	ModelName  string
	Quant      string
	NCtx       int
	NThreads   int
	NGpuLayers int
	HeadDim    int
	NumKVHeads int
	NumLayers  int
}

// Config holds all run parameters. Everything here is written into the JSON
// manifest so a run is fully reproducible.
type Config struct {
	N           int
	MaxGen      int
	Repeats     int
	Warmup      int
	Seed        int64
	PrefixShare float64
	EmbedDir    string
	Adapter     AdapterConfig
}

type harness struct {
	ad  Engine
	enc embedding.Encoder
	cfg Config
}

// Sample is one recorded fragment-path request outcome.
type Sample struct {
	Index      int     `json:"i"`
	ClusterID  int     `json:"cluster"`
	Status     string  `json:"status"` // EXACT | PARTIAL | MISS
	TTFTms     float64 `json:"ttft_ms"`
	InjectMs   float64 `json:"inject_ms"`
	FragBytes  int     `json:"frag_bytes"`
	FirstToken string  `json:"-"`
}

// Report is the full result payload (also serialised to JSON).
type Report struct {
	ColdTTFT      Dist     `json:"cold_ttft"`
	FragTTFTAll   Dist     `json:"frag_ttft_all"`  // includes misses (end-to-end)
	FragTTFTHits  Dist     `json:"frag_ttft_hits"` // hits only
	HitRatePct    float64  `json:"hit_rate_pct"`
	ExactHits     int      `json:"exact_hits"`
	PartialHits   int      `json:"partial_hits"`
	Misses        int      `json:"misses"`
	SpeedupAll    float64  `json:"speedup_all"`
	SpeedupHits   float64  `json:"speedup_hits"`
	SigDisjoint   bool     `json:"speedup_significant"` // CIs non-overlapping
	CorrectHits   int      `json:"correct_hits"`
	TotalHits     int      `json:"total_hits"`
	MatchRatePct  float64  `json:"match_rate_pct"`
	FragmentCount int      `json:"fragment_count"`
	FragTotalMB   float64  `json:"fragment_total_mb"`
	FragAvgMB     float64  `json:"fragment_avg_mb"`
	RSSDeltaMB    float64  `json:"rss_delta_mb"`
	Samples       []Sample `json:"-"` // written to CSV, not JSON manifest
}

func newHarness(cfg Config) (*harness, error) {
	ad, err := newAdapter(cfg.Adapter)
	if err != nil {
		return nil, err
	}
	enc, err := embedding.NewEncoder(cfg.EmbedDir)
	if err != nil {
		return nil, fmt.Errorf("embedding: %w", err)
	}
	return &harness{ad: ad, enc: enc, cfg: cfg}, nil
}

// timeMs runs fn Warmup times (discarded) then Repeats times (recorded) and
// returns the MEDIAN observed wall-clock in ms plus the last return value.
// Median is used per call-site to suppress scheduler jitter before the samples
// enter the aggregate distribution.
func (h *harness) timeMs(fn func() string) (float64, string) {
	for i := 0; i < h.cfg.Warmup; i++ {
		fn()
	}
	reps := h.cfg.Repeats
	if reps < 1 {
		reps = 1
	}
	obs := make([]float64, reps)
	var last string
	for i := 0; i < reps; i++ {
		t0 := time.Now()
		last = fn()
		obs[i] = float64(time.Since(t0).Microseconds()) / 1000.0
	}
	sort.Float64s(obs)
	return obs[len(obs)/2], last
}

// Run executes the two passes and assembles the report.
func (h *harness) Run(reqs []Request) *Report {
	ctx := context.Background()
	rep := &Report{}
	rssStart := readRSSMB()

	// ── Pass 1: COLD (no cache). Full prefill + 1 token = time-to-first-token. ──
	coldTTFT := make([]float64, len(reqs))
	coldFirst := make([]string, len(reqs))
	for i, r := range reqs {
		ms, first := h.timeMs(func() string {
			_ = h.ad.ClearKVCache(ctx)
			txt, _, _ := h.ad.Generate(ctx, r.Full, 0, 1)
			return txt
		})
		coldTTFT[i] = ms
		coldFirst[i] = first
	}

	// ── Pass 2: FRAGMENT (HNSW KV reuse). ──
	hnsw := core.NewHNSW(16, 50)
	storedVecs := map[int][]float32{}
	fragStore := map[int]*cache.KVFragment{}
	nextID := 1

	var fragAll, fragHits []float64
	var totalFragBytes int

	for i, r := range reqs {
		emb, err := h.enc.Encode(r.Full)
		if err != nil {
			// Embedding failure ⇒ treat as forced miss (cold), no store.
			ms, _ := h.timeMs(func() string {
				_ = h.ad.ClearKVCache(ctx)
				txt, _, _ := h.ad.Generate(ctx, r.Full, 0, 1)
				return txt
			})
			fragAll = append(fragAll, ms)
			rep.Misses++
			rep.Samples = append(rep.Samples, Sample{Index: i, ClusterID: r.ClusterID, Status: "MISS", TTFTms: ms})
			continue
		}

		bestSim, bestID := float32(-1), -1
		if len(storedVecs) > 0 {
			for _, nb := range hnsw.Search(emb, 1) {
				if v, ok := storedVecs[nb.ID]; ok {
					if s := cosine(emb, v); s > bestSim {
						bestSim, bestID = s, nb.ID
					}
				}
			}
		}

		switch {
		case bestID >= 0 && bestSim >= cache.SimilarityExact:
			s := h.hitSample(ctx, i, r, fragStore[bestID], "EXACT", coldFirst[i])
			fragAll = append(fragAll, s.TTFTms)
			fragHits = append(fragHits, s.TTFTms)
			rep.ExactHits++
			rep.Samples = append(rep.Samples, s)

		case bestID >= 0 && bestSim >= cache.SimilarityPartial:
			s := h.hitSample(ctx, i, r, fragStore[bestID], "PARTIAL", coldFirst[i])
			fragAll = append(fragAll, s.TTFTms)
			fragHits = append(fragHits, s.TTFTms)
			rep.PartialHits++
			rep.Samples = append(rep.Samples, s)

		default: // MISS: cold gen, then extract + store for future reuse.
			ms, _ := h.timeMs(func() string {
				_ = h.ad.ClearKVCache(ctx)
				txt, _, _ := h.ad.Generate(ctx, r.Full, 0, 1)
				return txt
			})
			fragAll = append(fragAll, ms)
			rep.Misses++
			s := Sample{Index: i, ClusterID: r.ClusterID, Status: "MISS", TTFTms: ms}

			if toks, err := h.ad.Tokenize(ctx, r.Full); err == nil && len(toks) >= cache.FragmentGranularityTokens {
				frag, err := h.ad.ExtractFragment(ctx, toks, 0, h.ad.ModelID().NumLayers, cache.FragmentLayerStride, emb)
				if err == nil && frag != nil {
					hnsw.Insert(nextID, emb)
					storedVecs[nextID] = emb
					fragStore[nextID] = frag
					nextID++
					totalFragBytes += frag.SizeBytes()
					s.FragBytes = frag.SizeBytes()
				}
			}
			rep.Samples = append(rep.Samples, s)
		}
	}

	// ── Aggregate ──
	rep.ColdTTFT = summarise(coldTTFT)
	rep.FragTTFTAll = summarise(fragAll)
	rep.FragTTFTHits = summarise(fragHits)

	n := len(reqs)
	hits := rep.ExactHits + rep.PartialHits
	if n > 0 {
		rep.HitRatePct = float64(hits) / float64(n) * 100
	}
	rep.SpeedupAll = speedup(rep.ColdTTFT, rep.FragTTFTAll)
	rep.SpeedupHits = speedup(rep.ColdTTFT, rep.FragTTFTHits)
	rep.SigDisjoint = ciDisjoint(rep.ColdTTFT, rep.FragTTFTAll)

	// Correctness: for hit requests, does the warm first token match the cold one?
	for _, s := range rep.Samples {
		if s.Status == "EXACT" || s.Status == "PARTIAL" {
			rep.TotalHits++
			if s.FirstToken == coldFirst[s.Index] {
				rep.CorrectHits++
			}
		}
	}
	if rep.TotalHits > 0 {
		rep.MatchRatePct = float64(rep.CorrectHits) / float64(rep.TotalHits) * 100
	}

	rep.FragmentCount = nextID - 1
	rep.FragTotalMB = float64(totalFragBytes) / (1024 * 1024)
	if rep.FragmentCount > 0 {
		rep.FragAvgMB = rep.FragTotalMB / float64(rep.FragmentCount)
	}
	rep.RSSDeltaMB = readRSSMB() - rssStart
	return rep
}

// hitSample measures inject + first-token generation for a cache hit.
func (h *harness) hitSample(ctx context.Context, i int, r Request, frag *cache.KVFragment, status, coldFirst string) Sample {
	var injMs float64
	injMs, _ = h.timeMs(func() string {
		_ = h.ad.ClearKVCache(ctx)
		_ = h.ad.InjectFragment(ctx, frag)
		return ""
	})
	genMs, first := h.timeMs(func() string {
		txt, _, _ := h.ad.Generate(ctx, r.Full, frag.TokenEnd, 1)
		return txt
	})
	return Sample{
		Index:      i,
		ClusterID:  r.ClusterID,
		Status:     status,
		TTFTms:     injMs + genMs,
		InjectMs:   injMs,
		FragBytes:  frag.SizeBytes(),
		FirstToken: first,
	}
}

// cosine similarity with explicit norms (embeddings may not be pre-normalised).
func cosine(a, b []float32) float32 {
	var dot, na, nb float32
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / float32(math.Sqrt(float64(na))*math.Sqrt(float64(nb)))
}

// readRSSMB reads resident set size from /proc/self/status (Linux/Android).
// Returns 0 on platforms without procfs.
func readRSSMB() float64 {
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "VmRSS:") {
			f := strings.Fields(line)
			if len(f) >= 2 {
				if kb, err := strconv.Atoi(f[1]); err == nil {
					return float64(kb) / 1024.0
				}
			}
		}
	}
	return 0
}
