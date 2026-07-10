//go:build !no_llamacpp

// Package adapter — llama.cpp backend implementation of KVAdapter.
//
// BUILD TAG: this file requires llama.h/libllama at compile time (see the
// #cgo directives below) and is included by default. If you only need the
// MLC or ONNX backends and don't have llama.cpp available, exclude it with
// `go build -tags no_llamacpp ./adapter/... ./sdk/android/...`
// (sdk/android/engine_llamacpp.go / engine_llamacpp_stub.go mirror this same
// tag so the JNI bridge compiles either way.)
//
// INTEGRATION NOTES FOR llama.cpp
// ─────────────────────────────────
// llama.cpp exposes KV cache access via:
//
//	llama_kv_cache_view_init()  — get a view of the current KV cache
//	llama_kv_cache_view_update() — refresh the view
//	llama_kv_cache_seq_rm()     — remove a sequence from the cache
//	llama_kv_cache_seq_cp()     — copy a sequence slot
//	llama_kv_cache_seq_shift()  — shift token positions
//
// For direct tensor extraction (Keys/Values as raw floats), llama.cpp does NOT
// expose a public API as of v0.0.3116. Two workarounds exist:
//
//  1. PATCH APPROACH: Add a thin C shim (extract_kv_slice.c) that reads
//     `ctx->kv_self.k_l[layer]` and `ctx->kv_self.v_l[layer]` directly.
//     Fragile but zero-overhead. Used in production for on-device Android builds.
//
//  2. GGML TENSOR APPROACH: After llama_decode(), iterate ggml_backend_tensor_get()
//     on the named tensors "cache_k_l%d" and "cache_v_l%d". Cleaner API,
//     available since llama.cpp b3117.
//
// This adapter uses approach 2 (the GGML tensor API) because it is stable and
// works without patching the llama.cpp source tree.
//
// For the Android JNI binding (EdgeSyncLLM.kt), the C functions declared in
// this file are exposed via the existing JNI bridge. No new JNI glue needed.
//
// CGO BINDING NOTE
// ─────────────────
// This file uses CGO to call into llama.cpp. In a CGO-less build (Android NDK
// cross-compilation via gomobile), replace the CGO calls with the JNI bridge
// calls in sdk/android/EdgeSyncLLM.kt.
//
// To build with CGO on the host for benchmarking:
//
//	CGO_CFLAGS="-I/path/to/llama.cpp" CGO_LDFLAGS="-L/path/to/llama.cpp/build -lllama" go build
package adapter

// #cgo CFLAGS: -I../../llama.cpp/include -I../../llama.cpp/ggml/include
// #cgo LDFLAGS: -L../../llama.cpp/build -lllama -lm
//
// #include "llama.h"
// #include <stdlib.h>
// #include <string.h>
//
// // NOTE: EdgeSync no longer needs a patched llama.cpp. Sequence state is moved
// // through the public llama_state_seq_get_data / llama_state_seq_set_data API,
// // which carries cell metadata (positions, sequence ids) as well as tensors.
// // The old edgesync_* C bridge is therefore gone: this links against vanilla
// // libllama, and survives llama.cpp upgrades.
import "C"

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"unsafe"

	"github.com/bossandboss/EdgeSync-LLM/cache"
)

// LlamaCppAdapter implements KVAdapter for the llama.cpp backend.
// One instance per loaded model context — do not share across goroutines.
type LlamaCppAdapter struct {
	ctx     *C.struct_llama_context // llama.cpp context pointer
	modelID cache.ModelID
}

// NewLlamaCppAdapter wraps an existing llama.cpp context.
// The model must already be loaded via llama_load_model_from_file() +
// llama_new_context_with_params() before calling this constructor.
//
// In CGO-less Android builds, pass nil for ctx and override the tensor
// extraction methods with JNI calls in EdgeSyncLLM.kt.
func NewLlamaCppAdapter(ctx unsafe.Pointer, modelID cache.ModelID) *LlamaCppAdapter {
	return &LlamaCppAdapter{
		ctx:     (*C.struct_llama_context)(ctx),
		modelID: modelID,
	}
}

// LoadLlamaCppModel loads a GGUF model from disk and returns a ready-to-use
// LlamaCppAdapter. This is the piece that was previously missing end-to-end:
// nativeInitialize() in sdk/android/jni_bridge.go accepted an "engine" string
// but never actually called this — globalAdapter stayed nil forever, so every
// inference-path JNI call silently no-opped via its "adapter not initialized"
// branch, even after a "successful" nativeInitialize().
//
// ⚠ NOT COMPILED OR VERIFIED IN THE ENVIRONMENT THAT WROTE THIS. This sandbox
// has no llama.h / no llama.cpp build available (see cache/differential.go
// and friends, which WERE build+test verified — this file could not be).
// Build this with the real headers (CGO_CFLAGS/CGO_LDFLAGS pointing at a
// compiled llama.cpp checkout, matching the #cgo directives at the top of
// this file) and fix any signature drift against your actual llama.cpp
// version (the exact fields of llama_model_params / llama_context_params
// change across llama.cpp releases — verify against the llama.h you're
// linking against, not just this comment).
//
// nThreads: CPU threads for generation. nGpuLayers: 0 = CPU-only inference;
// increase if a GPU/NPU backend is compiled into your libllama build.
func LoadLlamaCppModel(ggufPath string, modelID cache.ModelID, nCtx, nThreads, nGpuLayers int) (*LlamaCppAdapter, error) {
	cGgufPath := C.CString(ggufPath)
	defer C.free(unsafe.Pointer(cGgufPath))

	// llama_backend_init() is safe to call multiple times in recent llama.cpp
	// versions, but if your version asserts on double-init, guard this with a
	// sync.Once at package scope instead of calling it per-model-load.
	C.llama_backend_init()

	modelParams := C.llama_model_default_params()
	modelParams.n_gpu_layers = C.int32_t(nGpuLayers)

	model := C.llama_load_model_from_file(cGgufPath, modelParams)
	if model == nil {
		return nil, fmt.Errorf("LoadLlamaCppModel: llama_load_model_from_file failed for %q — check the path and that the file is a valid GGUF", ggufPath)
	}

	ctxParams := C.llama_context_default_params()
	ctxParams.n_ctx = C.uint32_t(nCtx)
	ctxParams.n_threads = C.int32_t(nThreads)
	ctxParams.n_threads_batch = C.int32_t(nThreads)

	ctx := C.llama_init_from_model(model, ctxParams)
	if ctx == nil {
		C.llama_free_model(model)
		return nil, fmt.Errorf("LoadLlamaCppModel: llama_init_from_model failed for %q (n_ctx=%d, n_threads=%d)", ggufPath, nCtx, nThreads)
	}

	return &LlamaCppAdapter{
		ctx:     ctx,
		modelID: modelID,
	}, nil
}

// ── KVAdapter identity ────────────────────────────────────────────────────────

func (a *LlamaCppAdapter) EngineName() string     { return llamaCppEngineName }
func (a *LlamaCppAdapter) EngineVersion() string  { return llamaCppEngineVersion }
func (a *LlamaCppAdapter) ModelID() cache.ModelID { return a.modelID }

// CompatibleWith: llama.cpp can self-inject only.
// Cross-engine reuse is not declared here (ONNX can read us, not the other way).
func (a *LlamaCppAdapter) CompatibleWith() []string { return []string{} }

// ── Fragment extraction ───────────────────────────────────────────────────────

// ExtractFragment runs a prefill decode on tokenIDs and captures the resulting
// KV tensors for layers [layerStart, layerEnd) with the given stride.
//
// The tensor data is serialized as flat IEEE 754 float32 in row-major order:
//
//	layout: [layer_index][kv_head][token][head_dim]
//
// This format is the canonical "llamacpp" serialization read by InjectFragment
// and by any adapter that declares "llamacpp" in CompatibleWith().
func (a *LlamaCppAdapter) ExtractFragment(
	ctx context.Context,
	tokenIDs []int32,
	layerStart, layerEnd, layerStride int,
	embedding []float32,
) (*cache.KVFragment, error) {
	if a.ctx == nil {
		return nil, fmt.Errorf("llamacpp: context is nil (CGO-less build?)")
	}

	tokenCount := len(tokenIDs)
	model := a.modelID

	// Count layers to capture
	var layersCaptured []int
	for l := layerStart; l < layerEnd; l += layerStride {
		layersCaptured = append(layersCaptured, l)
	}

	// ── Explicit prefill. ──
	// Decode the tokens into a cleared cache so the sequence provably holds
	// exactly tokenIDs[0:tokenCount] at positions 0..n-1.
	if err := a.prefill(ctx, tokenIDs); err != nil {
		return nil, fmt.Errorf("llamacpp ExtractFragment: prefill: %w", err)
	}

	// ── Serialise the sequence state via llama.cpp's PUBLIC API. ──
	//
	// WHY NOT RAW TENSOR SURGERY. Copying the K/V tensors moves the numbers but
	// not the bookkeeping: llama.cpp tracks, per cell, a position and a sequence
	// id, and builds its attention mask from those. A raw write leaves the cell
	// table empty, so the cache still believes the sequence is empty; attention
	// never sees the injected cells and the next decode reallocates over them.
	// The result is a fragment that is silently INERT — fast, because the prefix
	// is skipped, and wrong, because the prefix is gone.
	//
	// llama_state_seq_get_data serialises cells AND metadata, for every layer.
	// It also needs no fork of llama.cpp, which is what makes this deployable.
	//
	// CONSEQUENCE: the state blob is whole-sequence and all-layer. Per-layer
	// striding is not expressible here — and it was never sound anyway: skipping
	// layers leaves those layers with no KV for the prefix, so their attention
	// reads empty cells and the output cannot match the uncached path.
	blobSize := C.llama_state_seq_get_size(a.ctx, C.llama_seq_id(0))
	if blobSize == 0 {
		return nil, fmt.Errorf("llamacpp ExtractFragment: llama_state_seq_get_size returned 0")
	}
	blob := make([]byte, int(blobSize))
	written := C.llama_state_seq_get_data(
		a.ctx,
		(*C.uint8_t)(unsafe.Pointer(&blob[0])),
		blobSize,
		C.llama_seq_id(0),
	)
	if written == 0 {
		return nil, fmt.Errorf("llamacpp ExtractFragment: llama_state_seq_get_data failed")
	}
	blob = blob[:int(written)]

	// KVFragment carries two byte slices and rejects empty ones. The state blob
	// is a single opaque buffer, so it lives in Keys; Values holds a sentinel
	// marking the encoding, so a fragment written by the raw-tensor path can
	// never be fed to the state-API path (or vice versa) without being caught.
	keysBytes := blob
	valsBytes := []byte(stateBlobSentinel)

	fragmentID := generateFragmentID(tokenIDs, model)

	return cache.NewFragment(
		fragmentID,
		model,
		0, tokenCount,
		0, model.NumLayers, 1, // state blob covers every layer; stride is meaningless

		keysBytes, valsBytes,
		tokenIDs,
		embedding,
		llamaCppEngineName,
		llamaCppEngineVersion,
		cache.DefaultTTLSession,
	)
}

// InjectFragment loads tensor data from fragment into llama.cpp's active KV cache.
// After a successful inject, the engine can start generation from fragment.TokenEnd.
func (a *LlamaCppAdapter) InjectFragment(ctx context.Context, fragment *cache.KVFragment) error {
	if err := CanInject(a, fragment); err != nil {
		return fmt.Errorf("llamacpp InjectFragment: %w", err)
	}

	// Restore the sequence state produced by ExtractFragment. This writes the
	// cells AND their positions/sequence ids, so attention actually sees the
	// prefix — the property a raw tensor write cannot provide.
	if string(fragment.Values) != stateBlobSentinel {
		return fmt.Errorf("llamacpp InjectFragment: fragment is not a sequence-state blob (sentinel mismatch) — it was produced by an incompatible extractor")
	}
	if len(fragment.Keys) == 0 {
		return fmt.Errorf("llamacpp InjectFragment: empty state blob")
	}

	// llama_state_seq_set_data expects the destination sequence to be empty; the
	// blob carries its own cell positions. Clearing here makes the call
	// idempotent and keeps warm runs reproducible across repeats.
	if err := a.ClearKVCache(ctx); err != nil {
		return fmt.Errorf("llamacpp InjectFragment: clear: %w", err)
	}

	read := C.llama_state_seq_set_data(
		a.ctx,
		(*C.uint8_t)(unsafe.Pointer(&fragment.Keys[0])),
		C.size_t(len(fragment.Keys)),
		C.llama_seq_id(0),
	)
	// Contract: positive = ok, zero = failed to load.
	if read == 0 {
		return fmt.Errorf("llamacpp InjectFragment: llama_state_seq_set_data rejected a %d-byte blob — most often a model/context mismatch (n_ctx, layer count, head dims) between the fragment and this context", len(fragment.Keys))
	}
	return nil
}

// vocab returns the model's vocabulary handle. Modern llama.cpp moved tokenizer
// entry points off llama_model onto llama_vocab, reached via
// llama_get_model(ctx) -> llama_model_get_vocab(model).
// stateBlobSentinel marks a KVFragment whose Keys hold an opaque
// llama_state_seq_get_data blob rather than raw K/V tensors.
const stateBlobSentinel = "EDGESYNC-SEQSTATE-v1"

func (a *LlamaCppAdapter) vocab() *C.struct_llama_vocab {
	model := C.llama_get_model(a.ctx)
	if model == nil {
		return nil
	}
	return C.llama_model_get_vocab(model)
}

// Generate runs token generation from startTokenPos using the active KV cache.
// If a fragment was injected before calling Generate, set startTokenPos = fragment.TokenEnd.
//
// SEMANTICS. startTokenPos is the number of prompt tokens whose KV state is
// ALREADY present in the cache:
//
//	startTokenPos == 0            → cold path: prefill the entire prompt.
//	startTokenPos == frag.TokenEnd → warm path: prefill only prompt[startTokenPos:],
//	                                 the suffix the injected fragment doesn't cover.
//
// That skipped prefill is exactly the latency EdgeSync claims to remove, so this
// function is what makes the TTFT benchmark meaningful.
//
// Positions are assigned absolutely (pos = startTokenPos + i) so the suffix lines
// up with the injected fragment's KV cells rather than restarting at 0.
func (a *LlamaCppAdapter) Generate(
	ctx context.Context,
	prompt string,
	startTokenPos int,
	maxTokens int,
) (string, int, error) {
	if a.ctx == nil {
		return "", 0, fmt.Errorf("llamacpp: context is nil")
	}
	if maxTokens <= 0 {
		return "", 0, fmt.Errorf("llamacpp Generate: maxTokens must be > 0, got %d", maxTokens)
	}

	promptTokens, err := a.Tokenize(ctx, prompt)
	if err != nil {
		return "", 0, fmt.Errorf("llamacpp Generate: tokenize: %w", err)
	}
	if startTokenPos < 0 || startTokenPos > len(promptTokens) {
		return "", 0, fmt.Errorf("llamacpp Generate: startTokenPos %d out of range [0,%d]",
			startTokenPos, len(promptTokens))
	}

	suffix := promptTokens[startTokenPos:]
	if len(suffix) == 0 {
		// The fragment covers the whole prompt. We still need one token to
		// produce logits, so re-decode the final prompt token at its own
		// position. Guard against an empty prompt.
		if len(promptTokens) == 0 {
			return "", 0, fmt.Errorf("llamacpp Generate: empty prompt")
		}
		startTokenPos = len(promptTokens) - 1
		suffix = promptTokens[startTokenPos:]
	}

	nCtx := int(C.llama_n_ctx(a.ctx))
	if startTokenPos+len(suffix)+maxTokens > nCtx {
		return "", 0, fmt.Errorf("llamacpp Generate: prompt(%d)+gen(%d) exceeds n_ctx(%d)",
			startTokenPos+len(suffix), maxTokens, nCtx)
	}

	// ── Prefill: one batch holding the uncached suffix. ──
	batch := C.llama_batch_init(C.int32_t(len(suffix)), 0, 1)
	defer C.llama_batch_free(batch)

	fillBatch := func(tokens []int32, basePos int) {
		batch.n_tokens = C.int32_t(len(tokens))
		for i, tok := range tokens {
			setBatchToken(batch, i, tok, basePos+i, false)
		}
		// Only the last token needs logits — it's the one we sample from.
		setBatchLogits(batch, len(tokens)-1, true)
	}

	fillBatch(suffix, startTokenPos)
	if rc := C.llama_decode(a.ctx, batch); rc != 0 {
		return "", 0, fmt.Errorf("llamacpp Generate: prefill llama_decode failed (code %d)", int(rc))
	}

	// ── Sampler: greedy (argmax). Deterministic on purpose. ──
	// The benchmark compares the cold-path and warm-path first token to prove a
	// KV fragment does not change the output. Any stochastic sampler would make
	// that correctness check meaningless, so do NOT swap in top_k/top_p/dist here
	// without also disabling the match assertion.
	sparams := C.llama_sampler_chain_default_params()
	smpl := C.llama_sampler_chain_init(sparams)
	if smpl == nil {
		return "", 0, fmt.Errorf("llamacpp Generate: llama_sampler_chain_init returned nil")
	}
	defer C.llama_sampler_free(smpl)
	C.llama_sampler_chain_add(smpl, C.llama_sampler_init_greedy())

	voc := a.vocab()
	if voc == nil {
		return "", 0, fmt.Errorf("llamacpp Generate: vocab is nil")
	}

	// ── Decode loop. ──
	var sb strings.Builder
	nGenerated := 0
	curPos := startTokenPos + len(suffix)

	for i := 0; i < maxTokens; i++ {
		if err := ctx.Err(); err != nil {
			return sb.String(), nGenerated, err
		}

		// Sample from the logits of the last decoded token (idx -1).
		tok := C.llama_sampler_sample(smpl, a.ctx, C.int32_t(-1))

		// A sampled end-of-generation token IS a produced token: the prefill ran
		// and the model committed to an output. Time-to-first-token must count
		// it, otherwise a prompt whose greedy continuation is EOG reports zero
		// tokens and the caller times a decode that "did nothing" — which is how
		// an impossibly fast TTFT gets recorded. Count first, then stop.
		piece, err := a.tokenToPiece(voc, tok)
		if err != nil {
			return sb.String(), nGenerated, err
		}
		sb.WriteString(piece)
		nGenerated++

		if bool(C.llama_vocab_is_eog(voc, tok)) {
			break
		}
		C.llama_sampler_accept(smpl, tok)

		if nGenerated >= maxTokens {
			break
		}

		// Feed the sampled token back in to advance the KV cache.
		batch.n_tokens = 1
		setBatchToken(batch, 0, int32(tok), curPos, true)
		curPos++
		if rc := C.llama_decode(a.ctx, batch); rc != 0 {
			return sb.String(), nGenerated, fmt.Errorf("llamacpp Generate: decode failed at pos %d (code %d)", curPos-1, int(rc))
		}
	}

	return sb.String(), nGenerated, nil
}

// tokenToPiece converts a single token to its text piece, growing the buffer if
// llama_token_to_piece reports it was too small (it returns -needed in that case).
// special=true so control/EOG tokens render as text ("<|im_end|>") instead of
// an empty string. The benchmark compares the cold and warm first token; if EOG
// rendered as "" the comparison would pass trivially and prove nothing.
func (a *LlamaCppAdapter) tokenToPiece(voc *C.struct_llama_vocab, tok C.llama_token) (string, error) {
	buf := make([]byte, 64)
	n := C.llama_token_to_piece(voc, tok, (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)), 0, C.bool(true))
	if n < 0 {
		buf = make([]byte, -int(n))
		n = C.llama_token_to_piece(voc, tok, (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)), 0, C.bool(true))
		if n < 0 {
			return "", fmt.Errorf("llamacpp: llama_token_to_piece failed for token %d (code %d)", int(tok), int(n))
		}
	}
	return string(buf[:int(n)]), nil
}

// Tokenize converts text to llama.cpp token IDs via llama_tokenize().
//
// Two-pass: llama_tokenize returns -needed when the output buffer is too small.
// add_special=true prepends BOS per the model's convention (matching how the
// prompt would be tokenized at inference time); parse_special=true so chat
// template control tokens are recognised rather than emitted as literal text.
func (a *LlamaCppAdapter) Tokenize(_ context.Context, text string) ([]int32, error) {
	if a.ctx == nil {
		return nil, fmt.Errorf("llamacpp: context is nil")
	}
	voc := a.vocab()
	if voc == nil {
		return nil, fmt.Errorf("llamacpp Tokenize: vocab is nil")
	}
	if text == "" {
		return []int32{}, nil
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	textLen := C.int32_t(len(text))

	// Upper bound: one token per byte, plus room for BOS/EOS.
	capacity := len(text) + 8
	toks := make([]C.llama_token, capacity)

	n := C.llama_tokenize(voc, cText, textLen,
		(*C.llama_token)(unsafe.Pointer(&toks[0])), C.int32_t(capacity),
		C.bool(true), C.bool(true))

	if n < 0 {
		// Buffer too small: llama_tokenize returned -(required length).
		capacity = -int(n)
		toks = make([]C.llama_token, capacity)
		n = C.llama_tokenize(voc, cText, textLen,
			(*C.llama_token)(unsafe.Pointer(&toks[0])), C.int32_t(capacity),
			C.bool(true), C.bool(true))
		if n < 0 {
			return nil, fmt.Errorf("llamacpp Tokenize: llama_tokenize failed (code %d)", int(n))
		}
	}

	out := make([]int32, int(n))
	for i := 0; i < int(n); i++ {
		out[i] = int32(toks[i])
	}
	return out, nil
}

// setBatchToken writes slot i of a llama_batch. The C arrays are raw pointers,
// so index them through unsafe.Slice rather than Go slice syntax.
// Each token is assigned to sequence 0.
func setBatchToken(batch C.struct_llama_batch, i int, tok int32, pos int, wantLogits bool) {
	tokens := unsafe.Slice((*C.llama_token)(unsafe.Pointer(batch.token)), i+1)
	tokens[i] = C.llama_token(tok)

	positions := unsafe.Slice((*C.llama_pos)(unsafe.Pointer(batch.pos)), i+1)
	positions[i] = C.llama_pos(pos)

	nSeqIDs := unsafe.Slice((*C.int32_t)(unsafe.Pointer(batch.n_seq_id)), i+1)
	nSeqIDs[i] = 1

	seqIDPtrs := unsafe.Slice((**C.llama_seq_id)(unsafe.Pointer(batch.seq_id)), i+1)
	seqIDs := unsafe.Slice((*C.llama_seq_id)(unsafe.Pointer(seqIDPtrs[i])), 1)
	seqIDs[0] = 0

	setBatchLogits(batch, i, wantLogits)
}

func setBatchLogits(batch C.struct_llama_batch, i int, want bool) {
	if i < 0 {
		return
	}
	logits := unsafe.Slice((*C.int8_t)(unsafe.Pointer(batch.logits)), i+1)
	var v C.int8_t
	if want {
		v = 1
	}
	logits[i] = v
}

// ClearKVCache removes all sequences from the llama.cpp KV cache.
func (a *LlamaCppAdapter) ClearKVCache(_ context.Context) error {
	if a.ctx == nil {
		return nil
	}
	// llama_kv_cache_clear(ctx) was renamed to llama_memory_clear(mem, data)
	// in current llama.cpp — it now operates on the llama_memory_t handle
	// rather than the context directly. data=true clears the underlying
	// buffers too, matching the old function's behavior.
	mem := C.llama_get_memory(a.ctx)
	C.llama_memory_clear(mem, C.bool(true))
	return nil
}

// Close releases the llama.cpp context and its underlying model.
// NOTE: previously this only freed the context (llama_free), leaking the
// model handle from llama_load_model_from_file(). Only call llama_get_model
// + llama_free_model here if this adapter "owns" the model (i.e. it was
// created via LoadLlamaCppModel, not via NewLlamaCppAdapter wrapping an
// externally-managed context) — otherwise you'll double-free a model someone
// else is still responsible for. Track ownership explicitly if you use both
// construction paths in the same process.
func (a *LlamaCppAdapter) Close() error {
	if a.ctx == nil {
		return nil
	}
	model := C.llama_get_model(a.ctx)
	C.llama_free(a.ctx)
	if model != nil {
		C.llama_free_model(model)
	}
	a.ctx = nil
	return nil
}

// ── Serialization helpers ─────────────────────────────────────────────────────

// bytesToFloat32Slice deserializes the canonical llamacpp wire format.
func bytesToFloat32Slice(src []byte) []float32 {
	n := len(src) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(src[i*4:])
		out[i] = math.Float32frombits(bits)
	}
	return out
}

// prefill decodes tokenIDs at absolute positions 0..n-1 into a freshly cleared
// KV cache, so the cache provably holds exactly this prompt before extraction.
func (a *LlamaCppAdapter) prefill(ctx context.Context, tokenIDs []int32) error {
	if len(tokenIDs) == 0 {
		return fmt.Errorf("empty token list")
	}
	if err := a.ClearKVCache(ctx); err != nil {
		return err
	}
	batch := C.llama_batch_init(C.int32_t(len(tokenIDs)), 0, 1)
	defer C.llama_batch_free(batch)

	batch.n_tokens = C.int32_t(len(tokenIDs))
	for i, tok := range tokenIDs {
		setBatchToken(batch, i, tok, i, false)
	}
	setBatchLogits(batch, len(tokenIDs)-1, true)

	if rc := C.llama_decode(a.ctx, batch); rc != 0 {
		return fmt.Errorf("llama_decode failed (code %d)", int(rc))
	}
	return nil
}

// sliceCells extracts the first tokenCount cells from a whole-layer KV tensor.
//
// LAYOUT ASSUMPTION — READ BEFORE TRUSTING ANY SPEEDUP.
// llama.cpp stores each layer's K cache as [n_embd_k_gqa, n_cells], cell-major:
// cell i is a contiguous bytesPerCell run at offset i*bytesPerCell. The same
// holds for V *only when flash-attention is enabled*. With flash-attention
// disabled llama.cpp transposes V to [n_cells, n_embd_v_gqa], and a prefix of
// cells is then NOT contiguous — this function would silently return garbage.
//
// Transposition does not change the tensor's byte count, so no size check can
// detect it. The only arbiter is behavioural: after injecting a fragment, the
// first generated token must be identical to the cold path. The benchmark's
// first-token match rate exists precisely for this. If it drops below 100%,
// suspect V transposition before anything else.
func sliceCells(full []byte, nCells, tokenCount int, which string, layer int) ([]byte, error) {
	if len(full) == 0 {
		return nil, fmt.Errorf("llamacpp: layer %d %s tensor is empty", layer, which)
	}
	if len(full)%nCells != 0 {
		return nil, fmt.Errorf("llamacpp: layer %d %s tensor size %d is not divisible by %d cells — cache is not cell-major, slicing would corrupt the fragment",
			layer, which, len(full), nCells)
	}
	bytesPerCell := len(full) / nCells
	end := tokenCount * bytesPerCell
	if end > len(full) {
		return nil, fmt.Errorf("llamacpp: layer %d %s slice %d exceeds tensor %d", layer, which, end, len(full))
	}
	out := make([]byte, end)
	copy(out, full[:end])
	return out, nil
}
