//go:build no_llamacpp

package main

import (
	"fmt"

	"github.com/bossandboss/EdgeSync-LLM/adapter"
	"github.com/bossandboss/EdgeSync-LLM/cache"
)

// initLlamaCppAdapter stub for builds without llama.cpp (-tags no_llamacpp).
// See engine_llamacpp.go for the default implementation.
func initLlamaCppAdapter(ggufModelPath string, model cache.ModelID, nCtx, nThreads, nGpuLayers int) (adapter.KVAdapter, error) {
	return nil, fmt.Errorf("engine=llamacpp requested but this binary was built with -tags no_llamacpp (llama.cpp support not compiled in)")
}
