package main

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/bossandboss/EdgeSync-LLM/cache"
	"github.com/bossandboss/EdgeSync-LLM/core"
	"github.com/bossandboss/EdgeSync-LLM/embedding"
)

// AdapterConfig is consumed by the build-tagged newAdapter().
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

type Config struct {
	N           int
	MaxGen      int
	Repeats     int
	Warmup      int
	Seed        int64
	PrefixShare float64
	EmbedDir    string
	Strict      bool
	Adapter     AdapterConfig
}

type harness struct {
	ad  Engine
	enc embedding.Encoder
	cfg Config
	rng *rand.Rand

	// Error accounting. A benchmark that silently swallows engine errors will
	// report the latency of the FAILURE PATH as if it were a result. That is
	// how a 9 ms "TTFT" appears for a prefill that cannot cost less than a few
	// hundred ms. Every error is counted and surfaced.
	genErrs     int
	extractErrs int
	injectErrs  int
	firstErr    error
}

type Sample struct {
	Index      int     `json:"i"`
	ClusterID  int     `json:"cluster"`
	Status     string  `json:"status"`
	TTFTms     float64 `json:"ttft_ms"`
	ColdMs     float64 `json:"cold_ms"`
	InjectMs   float64 `json:"inject_ms"`
	FragBytes  int     `json:"frag_bytes"`
	FirstToken string  `json:"-"`
	ColdToken  string  `json:"-"`
	TruncToken string  `json:"-"`
}

type Report struct {
	ColdTTFT      Dist     `json:"cold_ttft"`
	FragTTFTAll   Dist     `json:"frag_ttft_all"`
	FragTTFTHits  Dist     `json:"frag_ttft_hits"`
	HitRatePct    float64  `json:"hit_rate_pct"`
	ExactHits     int      `json:"exact_hits"`
	PartialHits   int      `json:"partial_hits"`
	Misses        int      `json:"misses"`
	SpeedupAll    float64  `json:"speedup_all"`
	SpeedupHits   float64  `json:"speedup_hits"`
	SigDisjoint   bool     `json:"speedup_significant"`
	CorrectHits   int      `json:"correct_hits"`
	TotalHits     int      `json:"total_hits"`
	MatchRatePct  float64  `json:"match_rate_pct"`
	FragmentCount int      `json:"fragment_count"`
	FragTotalMB   float64  `json:"fragment_total_mb"`
	FragAvgMB     float64  `json:"fragment_avg_mb"`
	RSSDeltaMB    float64  `json:"rss_delta_mb"`
	GenErrors     int      `json:"generate_errors"`
	ExtractErrors int      `json:"extract_errors"`
	InjectErrors  int      `json:"inject_errors"`
	InertHits     int      `json:"inert_hits"`
	InertChecked  int      `json:"inert_checked"`
	PromptTokens  int      `json:"preflight_prompt_tokens"`
	PrefillTokPS  float64  `json:"preflight_prefill_tok_per_s"`
	Samples       []Sample `json:"-"`
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
	return &harness{ad: ad, enc: enc, cfg: cfg, rng: rand.New(rand.NewSource(cfg.Seed ^ 0x5eed))}, nil
}

// Preflight runs ONE request end to end with every error checked, and reports
// the prompt token count plus the implied prefill throughput. This is the
// sanity gate: if the engine is not really decoding, it surfaces here as an
// error or as an absurd tok/s, BEFORE any distribution is collected.
func (h *harness) Preflight(ctx context.Context, r Request) (int, float64, error) {
	toks, err := h.ad.Tokenize(ctx, r.Full)
	if err != nil {
		return 0, 0, fmt.Errorf("preflight Tokenize: %w", err)
	}
	if len(toks) == 0 {
		return 0, 0, fmt.Errorf("preflight: Tokenize returned 0 tokens - tokenizer not wired")
	}
	if err := h.ad.ClearKVCache(ctx); err != nil {
		return len(toks), 0, fmt.Errorf("preflight ClearKVCache: %w", err)
	}
	t0 := time.Now()
	out, n, err := h.ad.Generate(ctx, r.Full, 0, 1)
	el := time.Since(t0).Seconds()
	if err != nil {
		return len(toks), 0, fmt.Errorf("preflight Generate: %w", err)
	}
	if n < 1 {
		return len(toks), 0, fmt.Errorf("preflight: Generate produced %d tokens (expected >=1) - decode loop not running", n)
	}
	if strings.HasPrefix(out, "[llamacpp generation") {
		return len(toks), 0, fmt.Errorf("preflight: Generate returned the STUB placeholder - adapter not rebuilt")
	}
	tokPS := 0.0
	if el > 0 {
		tokPS = float64(len(toks)) / el
	}
	return len(toks), tokPS, nil
}

func (h *harness) timeMs(fn func() (string, error), kind string) (float64, string, error) {
	for i := 0; i < h.cfg.Warmup; i++ {
		if _, err := fn(); err != nil {
			return 0, "", fmt.Errorf("%s (warmup): %w", kind, err)
		}
	}
	reps := h.cfg.Repeats
	if reps < 1 {
		reps = 1
	}
	obs := make([]float64, reps)
	var last string
	for i := 0; i < reps; i++ {
		t0 := time.Now()
		s, err := fn()
		obs[i] = float64(time.Since(t0).Microseconds()) / 1000.0
		if err != nil {
			return 0, "", fmt.Errorf("%s: %w", kind, err)
		}
		last = s
	}
	sort.Float64s(obs)
	return obs[len(obs)/2], last, nil
}

func (h *harness) noteErr(err error, which *int) {
	*which++
	if h.firstErr == nil {
		h.firstErr = err
	}
}

func (h *harness) measureCold(ctx context.Context, r Request) (float64, string, error) {
	return h.timeMs(func() (string, error) {
		if err := h.ad.ClearKVCache(ctx); err != nil {
			return "", err
		}
		txt, _, err := h.ad.Generate(ctx, r.Full, 0, 1)
		return txt, err
	}, "cold Generate")
}

// measureWarm times the FULL warm path as one unit: clear, inject, generate.
//
// These cannot be timed as two independent repeated closures. Generate advances
// the KV cache (positions 123..132 for a 133-token prompt whose fragment covers
// 123). A second Generate starting again at 123 is then rejected by llama.cpp:
// "sequence positions must remain consecutive, Y = X + 1". Every repeat must
// therefore restore the cache first. Timing clear+inject+generate together is
// also the honest definition of warm TTFT: a deployment pays the injection cost
// on every cache hit.
func (h *harness) measureWarm(ctx context.Context, r Request, frag *cache.KVFragment) (float64, float64, string, error) {
	// Injection cost alone, for the reported breakdown.
	inj, _, err := h.timeMs(func() (string, error) {
		if e := h.ad.ClearKVCache(ctx); e != nil {
			return "", e
		}
		return "", h.ad.InjectFragment(ctx, frag)
	}, "InjectFragment")
	if err != nil {
		return 0, 0, "", err
	}
	// Warm TTFT: the whole path, re-established on every repeat.
	ttft, first, err := h.timeMs(func() (string, error) {
		if e := h.ad.ClearKVCache(ctx); e != nil {
			return "", e
		}
		if e := h.ad.InjectFragment(ctx, frag); e != nil {
			return "", e
		}
		txt, _, e := h.ad.Generate(ctx, r.Full, frag.TokenEnd, 1)
		return txt, e
	}, "warm Generate")
	if err != nil {
		return 0, 0, "", err
	}
	return ttft, inj, first, nil
}

// Run measures cold and warm for EACH request back to back, in randomised
// order, rather than one full cold pass followed by one full fragment pass.
// Sequential passes let slow drift (page cache warming, CPU frequency,
// thermals) masquerade as a speedup: a cold-then-warm layout reports a
// "significant" gain even when both paths execute identical code. Pairing and
// shuffling removes that confound.
func (h *harness) Run(reqs []Request) (*Report, error) {
	ctx := context.Background()
	rep := &Report{}
	rssStart := readRSSMB()

	hnsw := core.NewHNSW(16, 50)
	storedVecs := map[int][]float32{}
	fragStore := map[int]*cache.KVFragment{}
	nextID := 1

	var coldAll, fragAll, fragHits []float64
	var totalFragBytes int

	for i, r := range reqs {
		// Look up by the SHARED PREFIX, not the whole prompt. Two requests in the
		// same cluster differ in their final user turn; only their prefix KV state
		// is reusable.
		emb, encErr := h.enc.Encode(r.Prefix)

		bestSim, bestID := float32(-1), -1
		if encErr == nil && len(storedVecs) > 0 {
			for _, nb := range hnsw.Search(emb, 1) {
				if v, ok := storedVecs[nb.ID]; ok {
					if s := cosine(emb, v); s > bestSim {
						bestSim, bestID = s, nb.ID
					}
				}
			}
		}

		status := "MISS"
		if bestID >= 0 && bestSim >= cache.SimilarityExact {
			status = "EXACT"
		} else if bestID >= 0 && bestSim >= cache.SimilarityPartial {
			status = "PARTIAL"
		}

		if status == "MISS" {
			coldMs, coldTok, err := h.measureCold(ctx, r)
			if err != nil {
				h.noteErr(err, &h.genErrs)
				if h.cfg.Strict {
					return nil, fmt.Errorf("request %d: %w", i, err)
				}
				continue
			}
			coldAll = append(coldAll, coldMs)
			fragAll = append(fragAll, coldMs)
			rep.Misses++
			s := Sample{Index: i, ClusterID: r.ClusterID, Status: "MISS", TTFTms: coldMs, ColdMs: coldMs, ColdToken: coldTok}

			if encErr == nil {
				toks, err := h.ad.Tokenize(ctx, r.Full)
				if err != nil {
					h.noteErr(err, &h.extractErrs)
					if h.cfg.Strict {
						return nil, fmt.Errorf("request %d: Tokenize: %w", i, err)
					}
				} else if k := h.prefixTokenCount(ctx, r, toks); k >= cache.FragmentGranularityTokens {
					frag, err := h.ad.ExtractFragment(ctx, toks[:k], 0, h.ad.ModelID().NumLayers, cache.FragmentLayerStride, emb)
					if err != nil || frag == nil {
						if err == nil {
							err = fmt.Errorf("ExtractFragment returned nil fragment")
						}
						h.noteErr(err, &h.extractErrs)
						if h.cfg.Strict {
							return nil, fmt.Errorf("request %d: %w", i, err)
						}
					} else {
						hnsw.Insert(nextID, emb)
						storedVecs[nextID] = emb
						fragStore[nextID] = frag
						nextID++
						totalFragBytes += frag.SizeBytes()
						s.FragBytes = frag.SizeBytes()
					}
				}
			}
			rep.Samples = append(rep.Samples, s)
			continue
		}

		frag := fragStore[bestID]
		var coldMs, warmMs, injMs float64
		var coldTok, warmTok string
		var err error

		if h.rng.Intn(2) == 0 {
			coldMs, coldTok, err = h.measureCold(ctx, r)
			if err == nil {
				warmMs, injMs, warmTok, err = h.measureWarm(ctx, r, frag)
			}
		} else {
			warmMs, injMs, warmTok, err = h.measureWarm(ctx, r, frag)
			if err == nil {
				coldMs, coldTok, err = h.measureCold(ctx, r)
			}
		}
		if err != nil {
			h.noteErr(err, &h.injectErrs)
			if h.cfg.Strict {
				return nil, fmt.Errorf("request %d: %w", i, err)
			}
			continue
		}

		// Decisive control: does the warm output equal a no-prefix generation?
		truncTok, tErr := h.measureTruncated(ctx, r)
		if tErr != nil {
			truncTok = ""
		}

		coldAll = append(coldAll, coldMs)
		fragAll = append(fragAll, warmMs)
		fragHits = append(fragHits, warmMs)
		if status == "EXACT" {
			rep.ExactHits++
		} else {
			rep.PartialHits++
		}
		rep.Samples = append(rep.Samples, Sample{
			Index: i, ClusterID: r.ClusterID, Status: status,
			TTFTms: warmMs, ColdMs: coldMs, InjectMs: injMs,
			FragBytes: frag.SizeBytes(), FirstToken: warmTok, ColdToken: coldTok,
			TruncToken: truncTok,
		})
	}

	rep.ColdTTFT = summarise(coldAll)
	rep.FragTTFTAll = summarise(fragAll)
	rep.FragTTFTHits = summarise(fragHits)

	hits := rep.ExactHits + rep.PartialHits
	if n := len(reqs); n > 0 {
		rep.HitRatePct = float64(hits) / float64(n) * 100
	}
	rep.SpeedupAll = speedup(rep.ColdTTFT, rep.FragTTFTAll)
	rep.SpeedupHits = speedup(rep.ColdTTFT, rep.FragTTFTHits)
	rep.SigDisjoint = ciDisjoint(rep.ColdTTFT, rep.FragTTFTAll)

	for _, s := range rep.Samples {
		if s.Status == "EXACT" || s.Status == "PARTIAL" {
			rep.TotalHits++
			if s.FirstToken == s.ColdToken {
				rep.CorrectHits++
			}
			if s.TruncToken != "" {
				rep.InertChecked++
				// Warm agrees with the no-prefix control but not with cold:
				// the fragment demonstrably contributed nothing.
				if s.FirstToken == s.TruncToken && s.FirstToken != s.ColdToken {
					rep.InertHits++
				}
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
	rep.GenErrors = h.genErrs
	rep.ExtractErrors = h.extractErrs
	rep.InjectErrors = h.injectErrs
	return rep, nil
}

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

// prefixTokenCount returns how many leading tokens of the full prompt are
// covered byte-for-byte by r.Prefix.
//
// It does NOT simply return len(Tokenize(r.Prefix)). BPE merges across the join:
// the last token of the prefix, tokenized alone, can differ from the token that
// appears at that position when the full prompt is tokenized. Injecting a
// fragment whose final token disagrees with the prompt would corrupt generation
// in a way no size check catches. So compare the two token streams and keep only
// the leading run that agrees.
func (h *harness) prefixTokenCount(ctx context.Context, r Request, fullToks []int32) int {
	pt, err := h.ad.Tokenize(ctx, r.Prefix)
	if err != nil {
		return 0
	}
	n := len(pt)
	if n > len(fullToks) {
		n = len(fullToks)
	}
	k := 0
	for k < n && pt[k] == fullToks[k] {
		k++
	}
	return k
}

// measureTruncated generates from the suffix ALONE — the part of the prompt the
// fragment does not cover — with no injection and no prefix at all.
//
// This is the decisive control. Timing cannot tell a working fragment from an
// ignored one: both skip the prefix computation, so both are fast. Only the
// OUTPUT separates them.
//
//	warm token == cold token       -> the fragment carried the prefix's state.
//	warm token == truncated token  -> the prefix was never attended to; the
//	                                  fragment is inert and the speedup is an
//	                                  artifact of silently dropping context.
//
// A first-token match rate alone is weak here: a chat-tuned model asked to
// continue raw text often emits its end-of-turn token regardless of context, so
// cold and warm can agree by coincidence. Comparing against the truncated
// control removes that coincidence.
func (h *harness) measureTruncated(ctx context.Context, r Request) (string, error) {
	suffix := strings.TrimPrefix(r.Full, r.Prefix)
	if suffix == r.Full || suffix == "" {
		return "", fmt.Errorf("truncated control: prefix is not a prefix of full prompt")
	}
	if err := h.ad.ClearKVCache(ctx); err != nil {
		return "", err
	}
	txt, _, err := h.ad.Generate(ctx, suffix, 0, 1)
	return txt, err
}
