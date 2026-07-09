//go:build !realdevice

package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/bossandboss/EdgeSync-LLM/cache"
)

// engineReal is false for the host build. main.go reads this and stamps every
// output as SIMULATED so a mock run can NEVER be mistaken for a device
// measurement. This is the single most important line in the file: it is the
// guardrail that stops the "Expected results" problem from recurring.
const engineReal = false

// mockAdapter implements adapter.KVAdapter WITHOUT a real model. Its only job
// is to exercise the harness's control flow (tokenize → extract → inject →
// generate → stats → output) end to end so the harness itself is verified.
//
// Its Generate() sleeps for a tiny, explicitly-placeholder duration that is
// PROPORTIONAL TO THE NUMBER OF TOKENS IT ACTUALLY PROCESSES (full prompt when
// cold, only the tail after startTokenPos when warm). This makes the warm path
// measurably cheaper than the cold path — which proves the harness correctly
// discriminates the two — but it is NOT a hardware model and must never be
// quoted as a result. The SIMULATED banner enforces that.
type mockAdapter struct {
	model cache.ModelID
}

func newAdapter(_ AdapterConfig) (Engine, error) {
	return &mockAdapter{
		model: cache.ModelID{
			Architecture:  "qwen",
			Name:          "Qwen2.5-0.5B(MOCK)",
			Quantization:  "Q4_K_M",
			ContextLength: 4096,
			HeadDim:       64,
			NumKVHeads:    8,
			NumLayers:     24,
		},
	}, nil
}

func (a *mockAdapter) EngineName() string                 { return "mock" }
func (a *mockAdapter) EngineVersion() string              { return "0.0.0-placeholder" }
func (a *mockAdapter) ModelID() cache.ModelID             { return a.model }
func (a *mockAdapter) CompatibleWith() []string           { return nil }
func (a *mockAdapter) ClearKVCache(context.Context) error { return nil }
func (a *mockAdapter) Close() error                       { return nil }

// Tokenize: whitespace split. Deterministic, cheap. Real engine uses the model
// tokenizer; token counts differ but the harness only needs a stable integer.
func (a *mockAdapter) Tokenize(_ context.Context, text string) ([]int32, error) {
	fields := strings.Fields(text)
	ids := make([]int32, len(fields))
	for i := range fields {
		ids[i] = int32((len(fields[i]) * 131) % 32000)
	}
	return ids, nil
}

// placeholder per-token costs — NOT measured, NOT a claim.
const (
	mockPrefillPerTok = 40 * time.Microsecond
	mockDecodePerTok  = 120 * time.Microsecond
	mockInjectPerTok  = 3 * time.Microsecond
)

func (a *mockAdapter) Generate(_ context.Context, prompt string, startTokenPos, maxTokens int) (string, int, error) {
	toks := len(strings.Fields(prompt))
	prefillToks := toks - startTokenPos
	if prefillToks < 0 {
		prefillToks = 0
	}
	// Cold: prefill the whole prompt. Warm: prefill only the tail past the
	// injected fragment. Then decode maxTokens.
	time.Sleep(time.Duration(prefillToks)*mockPrefillPerTok + time.Duration(maxTokens)*mockDecodePerTok)
	return "[mock output]", maxTokens, nil
}

func (a *mockAdapter) ExtractFragment(
	_ context.Context,
	tokenIDs []int32,
	layerStart, layerEnd, layerStride int,
	embedding []float32,
) (*cache.KVFragment, error) {
	span := len(tokenIDs)
	if span < cache.FragmentGranularityTokens {
		return nil, fmt.Errorf("prefix too short to fragment: %d < %d", span, cache.FragmentGranularityTokens)
	}
	if span > cache.FragmentMaxTokenSpan {
		span = cache.FragmentMaxTokenSpan
		tokenIDs = tokenIDs[:span]
	}
	covered := (layerEnd - layerStart + layerStride - 1) / layerStride
	// Real KV bytes: covered × KVheads × span × headDim × 4 (float32), for K and V.
	perTensor := covered * a.model.NumKVHeads * span * a.model.HeadDim * 4
	keys := make([]byte, perTensor)
	values := make([]byte, perTensor)

	frag, err := cache.NewFragment(
		fmt.Sprintf("mock-%d-%d", tokenIDs[0], span),
		a.model,
		0, span,
		layerStart, layerEnd, layerStride,
		keys, values,
		tokenIDs[:span],
		embedding,
		a.EngineName(), a.EngineVersion(),
		cache.DefaultTTLSession,
	)
	return frag, err
}

// InjectFragment: cost proportional to fragment token span (memory copy).
func (a *mockAdapter) InjectFragment(_ context.Context, fragment *cache.KVFragment) error {
	time.Sleep(time.Duration(fragment.TokenSpan()) * mockInjectPerTok)
	return nil
}
