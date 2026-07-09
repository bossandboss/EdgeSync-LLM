package main

import (
	"context"

	"github.com/bossandboss/EdgeSync-LLM/cache"
)

// Engine is the subset of adapter.KVAdapter the benchmark drives. It is declared
// locally (Go interfaces are structural) so the HOST build never imports the
// adapter package — whose llama.cpp file needs llama.h/libllama at compile time.
// The real *adapter.LlamaCppAdapter satisfies this interface unchanged; the
// -tags realdevice build returns one from newAdapter().
type Engine interface {
	EngineName() string
	EngineVersion() string
	ModelID() cache.ModelID
	CompatibleWith() []string
	ExtractFragment(ctx context.Context, tokenIDs []int32, layerStart, layerEnd, layerStride int, embedding []float32) (*cache.KVFragment, error)
	InjectFragment(ctx context.Context, fragment *cache.KVFragment) error
	Generate(ctx context.Context, prompt string, startTokenPos, maxTokens int) (string, int, error)
	Tokenize(ctx context.Context, text string) ([]int32, error)
	ClearKVCache(ctx context.Context) error
	Close() error
}
