package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math/rand"
)

// Request is one benchmark request. Full is the exact text sent to the engine.
// PrefixTokensApprox is only used to decide fragment eligibility (span >= 64).
type Request struct {
	Full string
	// Prefix is the leading span of Full that is shared byte-for-byte with other
	// requests in the same cluster (system preamble + reused context block). KV
	// fragments are extracted over THIS span, never over Full: a fragment must
	// represent state that a later, different request can legitimately reuse.
	// Extracting over Full would bake in the varying user turn, and injecting it
	// for another request would silently generate from a KV cache that does not
	// correspond to that request's tokens.
	Prefix    string
	ClusterID int
}

// uniqueTag makes non-reuse preambles distinct from one another.
const uniqueTag = "ctx"

// buildCorpus generates a realistic on-device workload.
//
// WHY THIS SHAPE. KV-fragment reuse only pays off when requests SHARE a long
// prefix — a fixed system/instruction preamble plus a reused context block
// (RAG chunk, document, few-shot examples), followed by a short varying user
// turn. That is the assistant/RAG pattern EdgeSync targets. A corpus of
// unrelated one-off prompts would (correctly) show almost no benefit, so
// measuring on that would understate a real deployment; measuring on 100%
// identical prompts would overstate it. `prefixShare` controls the mix.
//
//	prefixShare = fraction of requests that reuse an existing shared prefix.
//	              The rest introduce a new prefix (cold path + fragment store).
func buildCorpus(n int, prefixShare float64, seed int64) []Request {
	rng := rand.New(rand.NewSource(seed))

	// A handful of distinct "conversation contexts". Each has a long shared
	// preamble (the part a fragment would cache) and a pool of short user turns.
	contexts := []struct {
		preamble  string
		userTurns []string
	}{
		{
			preamble: sysPreamble("a senior Go engineer reviewing embedded systems code") +
				ragBlock("llama.cpp KV cache", "ggml tensor layout", "ARM64 NEON kernels"),
			userTurns: []string{
				"summarise the risk in this diff",
				"is this allocation on the hot path",
				"how would you shrink the working set",
				"explain the lock ordering here",
				"what breaks under GQA",
			},
		},
		{
			preamble: sysPreamble("a French lycée teacher preparing conjugation drills") +
				ragBlock("verbes du 3e groupe", "subjonctif présent", "accords du participe passé"),
			userTurns: []string{
				"donne trois phrases d'exemple",
				"corrige cette réponse d'élève",
				"propose un exercice à trous",
				"explique la règle simplement",
				"liste les exceptions courantes",
			},
		},
		{
			preamble: sysPreamble("an Android performance engineer") +
				ragBlock("MediaPipe LiteRT-LM", "Gemma 3 1B on-device", "16KB memory page ABI"),
			userTurns: []string{
				"why did the model fail to load",
				"estimate the peak RSS",
				"which threads block the UI",
				"how to warm the cache at startup",
				"is this safe on a Cortex-A55",
			},
		},
		{
			preamble: sysPreamble("a retrieval assistant grounded in a fixed document") +
				ragBlock("write-ahead logging", "HNSW graph search", "cosine similarity recall"),
			userTurns: []string{
				"what does the document say about recall",
				"contrast WAL with rollback journalling",
				"when is efSearch too low",
				"give the complexity bound",
				"quote the relevant section",
			},
		},
	}

	reqs := make([]Request, n)
	// Track which contexts have already been "seen" so prefixShare is meaningful:
	// a request can only reuse a prefix that has appeared before.
	seen := make([]bool, len(contexts))
	for i := 0; i < n; i++ {
		cid := rng.Intn(len(contexts))
		reuse := rng.Float64() < prefixShare && seen[cid]
		ctx := contexts[cid]
		turn := ctx.userTurns[rng.Intn(len(ctx.userTurns))]

		var full, prefix string
		prefix = ctx.preamble
		if reuse {
			// Same preamble byte-for-byte (the cacheable prefix) + a different
			// short user turn. This is the request a KV fragment can serve.
			full = ctx.preamble + "\n\nUser: " + turn + "\nAssistant:"
		} else {
			// A genuinely NEW context: the preamble must differ, otherwise this
			// request shares a prefix with earlier ones and prefixShare has no
			// effect. A unique nonce inside the preamble guarantees a distinct
			// prefix, so -prefix-share 0 really does mean "no reuse possible"
			// (all misses, cold path only).
			seen[cid] = true
			prefix = ctx.preamble + fmt.Sprintf("\nSession: %s-%06d\n", uniqueTag, i)
			full = prefix + "\n\nUser: " + turn + "\nAssistant:"
		}
		reqs[i] = Request{Full: full, Prefix: prefix, ClusterID: cid}
	}
	return reqs
}

func sysPreamble(role string) string {
	return "System: You are " + role + ". Follow the instructions precisely, " +
		"cite the provided context, and never invent facts that are not supported " +
		"by the material below. Keep answers concise and technically exact."
}

func ragBlock(topics ...string) string {
	s := "\n\nContext:\n"
	for i, t := range topics {
		s += fmt.Sprintf("[%d] %s: reference notes, definitions, edge cases, and "+
			"worked examples covering %s in depth.\n", i+1, t, t)
	}
	return s
}

// corpusHash gives a stable fingerprint of the exact corpus used, so a run is
// reproducible and a reader can confirm two runs used identical inputs.
func corpusHash(reqs []Request) string {
	h := sha256.New()
	for _, r := range reqs {
		h.Write([]byte(r.Full))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}
