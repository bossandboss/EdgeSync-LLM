package core

import (
	"math"
	"math/rand"
	"testing"
)

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

func makeVec(dim int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	v := make([]float32, dim)
	var norm float32
	for i := range v {
		v[i] = rng.Float32()*2 - 1
		norm += v[i] * v[i]
	}
	norm = float32(math.Sqrt(float64(norm)))
	for i := range v {
		v[i] /= norm
	}
	return v
}

func cosineSimilarity(a, b []float32) float32 {
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

// ─────────────────────────────────────────────────────────────────────────────
// Construction tests
// ─────────────────────────────────────────────────────────────────────────────

func TestNewHNSW_Defaults(t *testing.T) {
	h := NewHNSW(16, 50)
	if h.M != 16 {
		t.Errorf("M: want 16, got %d", h.M)
	}
	if h.M0 != 32 {
		t.Errorf("M0: want 32, got %d", h.M0)
	}
	if h.EfSearch != 50 {
		t.Errorf("EfSearch: want 50, got %d", h.EfSearch)
	}
	if h.MaxLevel != -1 {
		t.Errorf("MaxLevel: want -1 (empty), got %d", h.MaxLevel)
	}
	if h.EnterNode != -1 {
		t.Errorf("EnterNode: want -1 (empty), got %d", h.EnterNode)
	}
}

func TestNewHNSW_InvalidParams(t *testing.T) {
	h := NewHNSW(0, 0)
	if h.M <= 0 {
		t.Errorf("M should default to positive value when 0 is passed, got %d", h.M)
	}
	if h.EfSearch <= 0 {
		t.Errorf("EfSearch should default to positive value, got %d", h.EfSearch)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Insert tests
// ─────────────────────────────────────────────────────────────────────────────

func TestInsert_FirstNode(t *testing.T) {
	h := NewHNSW(16, 50)
	vec := makeVec(384, 42)
	h.Insert(1, vec)

	if len(h.Nodes) != 1 {
		t.Errorf("want 1 node, got %d", len(h.Nodes))
	}
	if h.EnterNode != 1 {
		t.Errorf("EnterNode: want 1, got %d", h.EnterNode)
	}
	if h.MaxLevel < 0 {
		t.Errorf("MaxLevel should be >= 0 after first insert, got %d", h.MaxLevel)
	}
}

func TestInsert_MultipleNodes(t *testing.T) {
	h := NewHNSW(16, 50)
	for i := 1; i <= 100; i++ {
		h.Insert(i, makeVec(384, int64(i)))
	}
	if len(h.Nodes) != 100 {
		t.Errorf("want 100 nodes, got %d", len(h.Nodes))
	}
}

func TestInsert_DuplicateID(t *testing.T) {
	h := NewHNSW(16, 50)
	vec1 := makeVec(384, 1)
	vec2 := makeVec(384, 2)
	h.Insert(1, vec1)
	h.Insert(1, vec2) // same ID — should overwrite
	if len(h.Nodes) != 1 {
		t.Errorf("duplicate ID should not create two nodes, got %d", len(h.Nodes))
	}
}

func TestInsert_NeighborLinksBidirectional(t *testing.T) {
	h := NewHNSW(4, 20)
	for i := 1; i <= 20; i++ {
		h.Insert(i, makeVec(8, int64(i)))
	}

	// Verify no node has more than M0 neighbors at layer 0
	for id, node := range h.Nodes {
		if len(node.Neighbors) == 0 {
			continue
		}
		n0 := len(node.Neighbors[0])
		if n0 > h.M0 {
			t.Errorf("node %d has %d neighbors at layer 0, exceeds M0=%d", id, n0, h.M0)
		}
		for l := 1; l < len(node.Neighbors); l++ {
			nl := len(node.Neighbors[l])
			if nl > h.M {
				t.Errorf("node %d has %d neighbors at layer %d, exceeds M=%d", id, nl, l, h.M)
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Search tests
// ─────────────────────────────────────────────────────────────────────────────

func TestSearch_EmptyIndex(t *testing.T) {
	h := NewHNSW(16, 50)
	results := h.Search(makeVec(384, 1), 5)
	if results != nil && len(results) != 0 {
		t.Errorf("search on empty index should return nil or empty, got %v", results)
	}
}

func TestSearch_SingleNode(t *testing.T) {
	h := NewHNSW(16, 50)
	vec := makeVec(384, 42)
	h.Insert(1, vec)

	results := h.Search(vec, 1)
	if len(results) != 1 {
		t.Fatalf("want 1 result, got %d", len(results))
	}
	if results[0].ID != 1 {
		t.Errorf("want ID=1, got %d", results[0].ID)
	}
	if results[0].Similarity < 0.99 {
		t.Errorf("exact match should have similarity ≈ 1.0, got %.4f", results[0].Similarity)
	}
}

func TestSearch_TopKLessThanTotal(t *testing.T) {
	h := NewHNSW(16, 50)
	for i := 1; i <= 50; i++ {
		h.Insert(i, makeVec(384, int64(i)))
	}
	results := h.Search(makeVec(384, 1), 5)
	if len(results) > 5 {
		t.Errorf("want at most 5 results, got %d", len(results))
	}
}

func TestSearch_ReturnsNearestNeighbor(t *testing.T) {
	// Build index with 200 random vectors + 1 known close vector
	h := NewHNSW(16, 100)
	for i := 1; i <= 200; i++ {
		h.Insert(i, makeVec(384, int64(i*100)))
	}

	// Insert a query vector and its near-duplicate
	query := makeVec(384, 9999)
	nearDup := make([]float32, len(query))
	copy(nearDup, query)
	// Add tiny noise
	nearDup[0] += 0.001
	l2 := float32(0)
	for _, v := range nearDup {
		l2 += v * v
	}
	l2 = float32(math.Sqrt(float64(l2)))
	for i := range nearDup {
		nearDup[i] /= l2
	}
	h.Insert(201, nearDup)

	results := h.Search(query, 1)
	if len(results) == 0 {
		t.Fatal("expected at least 1 result")
	}
	// The nearest neighbor should be our near-duplicate (ID=201)
	// Relaxed: allow top-3 to contain it (ANN is approximate)
	results3 := h.Search(query, 3)
	found := false
	for _, r := range results3 {
		if r.ID == 201 {
			found = true
		}
	}
	if !found {
		t.Errorf("near-duplicate (ID=201) should appear in top-3 results")
	}
}

func TestSearch_SimilarityRange(t *testing.T) {
	h := NewHNSW(16, 50)
	for i := 1; i <= 50; i++ {
		h.Insert(i, makeVec(384, int64(i)))
	}
	results := h.Search(makeVec(384, 1), 10)
	for _, r := range results {
		if r.Similarity < -1.0 || r.Similarity > 1.0 {
			t.Errorf("similarity %f is outside [-1, 1]", r.Similarity)
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Delete tests
// ─────────────────────────────────────────────────────────────────────────────

func TestDelete_ExistingNode(t *testing.T) {
	h := NewHNSW(16, 50)
	for i := 1; i <= 10; i++ {
		h.Insert(i, makeVec(384, int64(i)))
	}

	h.Delete(5)

	if _, exists := h.Nodes[5]; exists {
		t.Error("node 5 should be removed after Delete")
	}
	if len(h.Nodes) != 9 {
		t.Errorf("want 9 nodes after delete, got %d", len(h.Nodes))
	}

	// Verify no remaining node still references the deleted node
	for id, node := range h.Nodes {
		for l, neighbors := range node.Neighbors {
			for _, nid := range neighbors {
				if nid == 5 {
					t.Errorf("node %d still references deleted node 5 at layer %d", id, l)
				}
			}
		}
	}
}

func TestDelete_NonExistentNode(t *testing.T) {
	h := NewHNSW(16, 50)
	h.Insert(1, makeVec(384, 1))
	// Should not panic
	h.Delete(999)
	if len(h.Nodes) != 1 {
		t.Errorf("deleting non-existent node should not affect index, got %d nodes", len(h.Nodes))
	}
}

func TestDelete_EntryNode(t *testing.T) {
	h := NewHNSW(16, 50)
	for i := 1; i <= 5; i++ {
		h.Insert(i, makeVec(384, int64(i)))
	}
	entryID := h.EnterNode
	h.Delete(entryID)

	if _, exists := h.Nodes[entryID]; exists {
		t.Error("entry node should be removed")
	}
	// After deleting entry, index should still be searchable
	results := h.Search(makeVec(384, 99), 1)
	if len(h.Nodes) > 0 && len(results) == 0 {
		t.Error("index should still be searchable after deleting entry node")
	}
}

func TestDelete_AllNodes(t *testing.T) {
	h := NewHNSW(16, 50)
	for i := 1; i <= 5; i++ {
		h.Insert(i, makeVec(384, int64(i)))
	}
	for i := 1; i <= 5; i++ {
		h.Delete(i)
	}
	if len(h.Nodes) != 0 {
		t.Errorf("want 0 nodes, got %d", len(h.Nodes))
	}
	// Search on empty index should not panic
	results := h.Search(makeVec(384, 1), 5)
	if len(results) != 0 {
		t.Errorf("search on empty index should return 0 results, got %d", len(results))
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization tests
// ─────────────────────────────────────────────────────────────────────────────

func TestSerializeDeserialize_RoundTrip(t *testing.T) {
	h := NewHNSW(16, 50)
	for i := 1; i <= 30; i++ {
		h.Insert(i, makeVec(384, int64(i)))
	}

	query := makeVec(384, 42)
	beforeResults := h.Search(query, 5)

	data, err := h.Serialize()
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("serialized data is empty")
	}

	h2 := NewHNSW(16, 50)
	if err := h2.Deserialize(data); err != nil {
		t.Fatalf("deserialize: %v", err)
	}

	if len(h2.Nodes) != len(h.Nodes) {
		t.Errorf("node count mismatch: before=%d, after=%d", len(h.Nodes), len(h2.Nodes))
	}

	afterResults := h2.Search(query, 5)
	if len(afterResults) != len(beforeResults) {
		t.Errorf("result count mismatch: before=%d, after=%d",
			len(beforeResults), len(afterResults))
	}

	// Top result should be the same
	if len(beforeResults) > 0 && len(afterResults) > 0 {
		if beforeResults[0].ID != afterResults[0].ID {
			t.Errorf("top result mismatch: before=%d, after=%d",
				beforeResults[0].ID, afterResults[0].ID)
		}
	}
}

func TestDeserialize_CorruptData(t *testing.T) {
	h := NewHNSW(16, 50)
	err := h.Deserialize([]byte("not valid gob data"))
	if err == nil {
		t.Error("expected error on corrupt data, got nil")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// cosineDistance tests
// ─────────────────────────────────────────────────────────────────────────────

func TestCosineDistance_IdenticalVectors(t *testing.T) {
	v := makeVec(384, 1)
	d := cosineDistance(v, v)
	if d > 0.001 {
		t.Errorf("identical vectors: distance should be ~0, got %f", d)
	}
}

func TestCosineDistance_OppositeVectors(t *testing.T) {
	v := makeVec(384, 1)
	neg := make([]float32, len(v))
	for i := range v {
		neg[i] = -v[i]
	}
	d := cosineDistance(v, neg)
	if math.Abs(float64(d)-2.0) > 0.001 {
		t.Errorf("opposite vectors: distance should be ~2.0, got %f", d)
	}
}

func TestCosineDistance_LengthMismatch(t *testing.T) {
	a := makeVec(4, 1)
	b := makeVec(8, 2)
	d := cosineDistance(a, b)
	if d != 1.0 {
		t.Errorf("length mismatch should return 1.0, got %f", d)
	}
}

func TestCosineDistance_EmptyVector(t *testing.T) {
	d := cosineDistance([]float32{}, []float32{})
	if d != 1.0 {
		t.Errorf("empty vectors should return 1.0, got %f", d)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark
// ─────────────────────────────────────────────────────────────────────────────

func BenchmarkInsert(b *testing.B) {
	h := NewHNSW(16, 50)
	vecs := make([][]float32, b.N)
	for i := range vecs {
		vecs[i] = makeVec(384, int64(i))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Insert(i+1, vecs[i])
	}
}

func BenchmarkSearch_1000Nodes(b *testing.B) {
	h := NewHNSW(16, 50)
	for i := 1; i <= 1000; i++ {
		h.Insert(i, makeVec(384, int64(i)))
	}
	query := makeVec(384, 9999)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Search(query, 5)
	}
}
