//go:build realdevice

// This file compiles ONLY with:  go build -tags realdevice ./benchmark/real/
// which requires CGO + the llama.cpp shared library on the target (Android
// ARM64 or a desktop with the C deps linked, per README § Building).
//
// It wires the REAL llama.cpp adapter, so every latency the harness records is
// a real prefill/inject/decode on a real model. engineReal = true, so the
// output is stamped MEASURED rather than SIMULATED.
package main

import (
	"fmt"

	"github.com/bossandboss/EdgeSync-LLM/adapter"
	"github.com/bossandboss/EdgeSync-LLM/cache"
)

const engineReal = true

func newAdapter(cfg AdapterConfig) (Engine, error) {
	if cfg.ModelPath == "" {
		return nil, fmt.Errorf("realdevice build requires -model-path pointing at a .gguf file")
	}
	// The ModelID MUST match the loaded GGUF (architecture, layer count, head
	// dims). Fragments are rejected on mismatch, so get this right for the model
	// you actually ship. Defaults below are Qwen2.5-0.5B / Q4_K_M; override via
	// flags if you benchmark a different model.
	modelID := cache.ModelID{
		Architecture:  cfg.Arch,
		Name:          cfg.ModelName,
		Quantization:  cfg.Quant,
		ContextLength: cfg.NCtx,
		HeadDim:       cfg.HeadDim,
		NumKVHeads:    cfg.NumKVHeads,
		NumLayers:     cfg.NumLayers,
	}
	a, err := adapter.LoadLlamaCppModel(cfg.ModelPath, modelID, cfg.NCtx, cfg.NThreads, cfg.NGpuLayers)
	if err != nil {
		return nil, fmt.Errorf("LoadLlamaCppModel: %w", err)
	}
	var e Engine = a
	return e, nil
}
