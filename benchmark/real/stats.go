// Package main — real (measured) EdgeSync-LLM benchmark.
//
// stats.go: distribution helpers. Everything here operates on OBSERVED
// wall-clock samples. No hardware constants, no cost formulas.
package main

import (
	"math"
	"sort"
)

// Dist summarises a slice of latency samples (milliseconds).
type Dist struct {
	N      int     `json:"n"`
	Mean   float64 `json:"mean_ms"`
	Median float64 `json:"median_ms"`
	P50    float64 `json:"p50_ms"`
	P90    float64 `json:"p90_ms"`
	P95    float64 `json:"p95_ms"`
	P99    float64 `json:"p99_ms"`
	Min    float64 `json:"min_ms"`
	Max    float64 `json:"max_ms"`
	Stddev float64 `json:"stddev_ms"`
	// CI95Lo/Hi: 95% confidence interval on the MEAN (normal approximation,
	// mean ± 1.96·stddev/sqrt(n)). Reported so a reader can judge whether an
	// observed difference is inside the noise floor.
	CI95Lo float64 `json:"ci95_lo_ms"`
	CI95Hi float64 `json:"ci95_hi_ms"`
}

func summarise(samples []float64) Dist {
	n := len(samples)
	if n == 0 {
		return Dist{}
	}
	s := make([]float64, n)
	copy(s, samples)
	sort.Float64s(s)

	var sum float64
	for _, v := range s {
		sum += v
	}
	mean := sum / float64(n)

	var sq float64
	for _, v := range s {
		d := v - mean
		sq += d * d
	}
	// sample stddev (n-1) when we have >1 point.
	var sd float64
	if n > 1 {
		sd = math.Sqrt(sq / float64(n-1))
	}
	sem := 0.0
	if n > 0 {
		sem = sd / math.Sqrt(float64(n))
	}

	return Dist{
		N:      n,
		Mean:   mean,
		Median: pct(s, 50),
		P50:    pct(s, 50),
		P90:    pct(s, 90),
		P95:    pct(s, 95),
		P99:    pct(s, 99),
		Min:    s[0],
		Max:    s[n-1],
		Stddev: sd,
		CI95Lo: mean - 1.96*sem,
		CI95Hi: mean + 1.96*sem,
	}
}

// pct returns the p-th percentile of an already-sorted slice using linear
// interpolation between closest ranks.
func pct(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return sorted[0]
	}
	rank := (p / 100.0) * float64(n-1)
	lo := int(math.Floor(rank))
	hi := int(math.Ceil(rank))
	if lo == hi {
		return sorted[lo]
	}
	frac := rank - float64(lo)
	return sorted[lo] + frac*(sorted[hi]-sorted[lo])
}

// speedup returns cold.Mean / warm.Mean, guarding against divide-by-zero.
// A speedup is only meaningful if the two CIs do not overlap; caller checks that.
func speedup(cold, warm Dist) float64 {
	if warm.Mean <= 0 {
		return 0
	}
	return cold.Mean / warm.Mean
}

// ciDisjoint reports whether the two 95% CIs on the mean are non-overlapping,
// i.e. the measured difference is unlikely to be pure noise.
func ciDisjoint(a, b Dist) bool {
	return a.CI95Hi < b.CI95Lo || b.CI95Hi < a.CI95Lo
}
