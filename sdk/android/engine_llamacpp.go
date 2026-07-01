//go:build !no_llamacpp

package main

import (
	"github.com/bossandboss/EdgeSync-LLM/adapter"
	"github.com/bossandboss/EdgeSync-LLM/cache"
)

// initLlamaCppAdapter is the default (llama.cpp available) implementation.
// See engine_llamacpp_stub.go for the -tags no_llamacpp counterpart.
func initLlamaCppAdapter(ggufModelPath string, model cache.ModelID, nCtx, nThreads, nGpuLayers int) (adapter.KVAdapter, error) {
	return adapter.LoadLlamaCppModel(ggufModelPath, model, nCtx, nThreads, nGpuLayers)
}
