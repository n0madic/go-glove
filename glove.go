// This package provides a GloVe model implementation for word vectorization.
// It includes functionality for training the model, saving/loading vectors,
// and performing word analogies.
package glove

import (
	"bufio"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// Hyperparameters from the paper
const (
	XMAX          = 100.0 // Cutoff parameter for the weighting function
	ALPHA         = 0.75  // Exponent for the weighting function (3/4)
	LEARNING_RATE = 0.05  // Initial learning rate
	MIN_COUNT     = 5     // Minimum word frequency to include in the vocabulary
	EPS           = 1e-8  // Small value for AdaGrad

	// Sharding for thread-safe parameter updates
	NumShards = 8192 // Power of 2 for fast bitwise operations
)

// CoocEntry represents an entry in the co-occurrence matrix
type CoocEntry struct {
	Word1 int     // Index of the first word
	Word2 int     // Index of the second word
	Count float64 // Co-occurrence count
}

// TrainingProgress contains information about the current training progress
type TrainingProgress struct {
	Iteration     int           // Current iteration (1-based)
	MaxIterations int           // Total number of iterations
	Cost          float64       // Current cost/loss value
	TimeElapsed   time.Duration // Time elapsed since training started
}

// ProgressCallback is a function type for receiving training progress updates
type ProgressCallback func(progress TrainingProgress)

// GloVe model
type GloVe struct {
	// Model parameters
	VectorSize   int  // Vector dimensionality
	Symmetric    bool // Symmetric context window
	MaxVocabSize int  // Maximum vocabulary size

	// Data
	Vocab     map[string]int // Mapping word -> index
	InvVocab  []string       // Mapping index -> word
	WordCount []int          // Word frequencies
	VocabSize int            // Vocabulary size
	Cooccur   []CoocEntry    // Co-occurrence matrix (sparse)

	// Model parameters
	W      [][]float64 // Word vector matrix
	WTilde [][]float64 // Context vector matrix
	B      []float64   // Biases for words
	BTilde []float64   // Biases for contexts

	// Gradients for AdaGrad
	GradSqW      [][]float64 // Accumulated squared gradients for W
	GradSqWTilde [][]float64 // Accumulated squared gradients for WTilde
	GradSqB      []float64   // Accumulated squared gradients for B
	GradSqBTilde []float64   // Accumulated squared gradients for BTilde

	// Synchronization for thread-safe parameter updates
	shardMutexes [NumShards]sync.Mutex
}

// NewGloVe creates a new GloVe model
func NewGloVe() *GloVe {
	return &GloVe{
		Symmetric:    true,
		MaxVocabSize: 400000, // As in the paper
		Vocab:        make(map[string]int),
	}
}

// BuildVocab builds the vocabulary from a corpus
func (g *GloVe) BuildVocab(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	// Count word frequencies (lowercased, whitespace tokenization)
	wordFreq := Tokenize(f, MIN_COUNT)

	if len(wordFreq) > g.MaxVocabSize {
		wordFreq = wordFreq[:g.MaxVocabSize]
	}

	g.VocabSize = len(wordFreq)
	g.InvVocab = make([]string, g.VocabSize)
	g.WordCount = make([]int, g.VocabSize)
	g.Vocab = make(map[string]int, g.VocabSize)

	for i, wf := range wordFreq {
		g.Vocab[wf.Word] = i
		g.InvVocab[i] = wf.Word
		g.WordCount[i] = wf.Freq
	}

	return nil
}

// helper to map words to vocab indices (skip OOV)
func indicesFromWords(words []string, vocab map[string]int) []int {
	idxs := make([]int, 0, len(words))
	for _, w := range words {
		if idx, ok := vocab[w]; ok {
			idxs = append(idxs, idx)
		}
	}
	return idxs
}

// pairKey creates a unique key for a word pair
func (g *GloVe) pairKey(i, j int) uint64 {
	return uint64(i)<<32 | uint64(j)
}

// unpairKey extracts indices from a key
func (g *GloVe) unpairKey(key uint64) (int, int) {
	return int(key >> 32), int(key & 0xFFFFFFFF)
}

func (g *GloVe) BuildCooccurrenceMatrix(filename string, windowSize int) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	coocMap := make(map[uint64]float64, 1<<20)

	scanner := bufio.NewScanner(f)
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		// Use the same tokenizer as BuildVocab for consistency
		lineReader := strings.NewReader(line)
		wordFreqs := Tokenize(lineReader, 1)

		// Extract just the words (ignore frequencies) and convert to indices
		tokens := make([]string, len(wordFreqs))
		for i, wf := range wordFreqs {
			tokens[i] = wf.Word
		}

		if len(tokens) == 0 {
			continue
		}
		indices := indicesFromWords(tokens, g.Vocab)
		for i := 0; i < len(indices); i++ {
			left := i - windowSize
			if left < 0 {
				left = 0
			}
			for j := left; j < i; j++ {
				idx1 := indices[i]
				idx2 := indices[j]
				dist := float64(i - j)
				if dist <= 0 {
					continue
				}
				// distance weighting 1/d as in the paper (Sec. 4.2)
				w := 1.0 / dist

				// one pass: add (i->j); if symmetric, also add (j->i)
				coocMap[g.pairKey(idx1, idx2)] += w
				if g.Symmetric {
					coocMap[g.pairKey(idx2, idx1)] += w
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}

	g.Cooccur = make([]CoocEntry, 0, len(coocMap))
	for key, count := range coocMap {
		w1, w2 := g.unpairKey(key)
		g.Cooccur = append(g.Cooccur, CoocEntry{Word1: w1, Word2: w2, Count: count})
	}

	return nil
}

// InitializeParameters initializes model parameters
func (g *GloVe) InitializeParameters(vectorSize int) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Set the vector size
	g.VectorSize = vectorSize

	// Initialize weights with random values from [-0.5, 0.5] / vectorSize
	initRange := 0.5 / float64(g.VectorSize)

	// Initialize matrices
	g.W = make([][]float64, g.VocabSize)
	g.WTilde = make([][]float64, g.VocabSize)
	g.B = make([]float64, g.VocabSize)
	g.BTilde = make([]float64, g.VocabSize)

	g.GradSqW = make([][]float64, g.VocabSize)
	g.GradSqWTilde = make([][]float64, g.VocabSize)
	g.GradSqB = make([]float64, g.VocabSize)
	g.GradSqBTilde = make([]float64, g.VocabSize)

	for i := 0; i < g.VocabSize; i++ {
		g.W[i] = make([]float64, g.VectorSize)
		g.WTilde[i] = make([]float64, g.VectorSize)
		g.GradSqW[i] = make([]float64, g.VectorSize)
		g.GradSqWTilde[i] = make([]float64, g.VectorSize)

		for j := 0; j < g.VectorSize; j++ {
			g.W[i][j] = (rng.Float64() - 0.5) * 2 * initRange
			g.WTilde[i][j] = (rng.Float64() - 0.5) * 2 * initRange
			g.GradSqW[i][j] = 1.0
			g.GradSqWTilde[i][j] = 1.0
		}

		g.B[i] = (rng.Float64() - 0.5) * 2 * initRange
		g.BTilde[i] = (rng.Float64() - 0.5) * 2 * initRange
		g.GradSqB[i] = 1.0
		g.GradSqBTilde[i] = 1.0
	}
}

// InitOptimizerState allocates AdaGrad accumulators if nil or mismatched.
// This is called after LoadModelState/LoadVectors to ensure training can resume.
func (g *GloVe) InitOptimizerState() {
	if g.VectorSize <= 0 || g.VocabSize <= 0 {
		return
	}

	// Initialize GradSqW if nil or mismatched
	if g.GradSqW == nil || len(g.GradSqW) != g.VocabSize {
		g.GradSqW = make([][]float64, g.VocabSize)
		for i := range g.GradSqW {
			g.GradSqW[i] = make([]float64, g.VectorSize)
			for k := range g.GradSqW[i] {
				g.GradSqW[i][k] = 1.0
			}
		}
	}

	// Initialize GradSqWTilde if nil or mismatched
	if g.GradSqWTilde == nil || len(g.GradSqWTilde) != g.VocabSize {
		g.GradSqWTilde = make([][]float64, g.VocabSize)
		for i := range g.GradSqWTilde {
			g.GradSqWTilde[i] = make([]float64, g.VectorSize)
			for k := range g.GradSqWTilde[i] {
				g.GradSqWTilde[i][k] = 1.0
			}
		}
	}

	// Initialize GradSqB if nil or mismatched
	if g.GradSqB == nil || len(g.GradSqB) != g.VocabSize {
		g.GradSqB = make([]float64, g.VocabSize)
		for i := range g.GradSqB {
			g.GradSqB[i] = 1.0
		}
	}

	// Initialize GradSqBTilde if nil or mismatched
	if g.GradSqBTilde == nil || len(g.GradSqBTilde) != g.VocabSize {
		g.GradSqBTilde = make([]float64, g.VocabSize)
		for i := range g.GradSqBTilde {
			g.GradSqBTilde[i] = 1.0
		}
	}
}

// withShardLocks executes a function with proper shard locking for words i and j
func (g *GloVe) withShardLocks(i, j int, fn func()) {
	iShard := i & (NumShards - 1)
	jShard := j & (NumShards - 1)

	if iShard == jShard {
		g.shardMutexes[iShard].Lock()
		defer g.shardMutexes[iShard].Unlock()
	} else {
		// Lock in order to prevent deadlock
		first, second := iShard, jShard
		if iShard > jShard {
			first, second = jShard, iShard
		}
		g.shardMutexes[first].Lock()
		defer g.shardMutexes[first].Unlock()
		g.shardMutexes[second].Lock()
		defer g.shardMutexes[second].Unlock()
	}

	fn()
}

// updateParameters performs AdaGrad updates for word pair (i, j)
func (g *GloVe) updateParameters(i, j int, scale, lr float64) {
	// Vector updates
	for k := 0; k < g.VectorSize; k++ {
		gradW := scale * g.WTilde[j][k]
		gradWT := scale * g.W[i][k]

		g.GradSqW[i][k] += gradW * gradW
		g.W[i][k] -= lr * gradW / math.Sqrt(g.GradSqW[i][k]+EPS)

		g.GradSqWTilde[j][k] += gradWT * gradWT
		g.WTilde[j][k] -= lr * gradWT / math.Sqrt(g.GradSqWTilde[j][k]+EPS)
	}

	// Bias updates
	g.GradSqB[i] += scale * scale
	g.B[i] -= lr * scale / math.Sqrt(g.GradSqB[i]+EPS)

	g.GradSqBTilde[j] += scale * scale
	g.BTilde[j] -= lr * scale / math.Sqrt(g.GradSqBTilde[j]+EPS)
}

// weightingFunction computes the weighting function f(x) from equation 9
func weightingFunction(x float64) float64 {
	if x < XMAX {
		return math.Pow(x/XMAX, ALPHA)
	}
	return 1.0
}

// Train trains the GloVe model
func (g *GloVe) Train(maxIter, numThreads int) {
	// Use TrainWithCallback with no callback for backward compatibility
	g.TrainWithCallback(maxIter, numThreads, nil)
}

// TrainWithCallback trains the GloVe model with optional progress callback
func (g *GloVe) TrainWithCallback(maxIter, numThreads int, callback ProgressCallback) {
	// Default iterations consistent with the paper
	if maxIter == 0 {
		if g.VectorSize < 300 {
			maxIter = 50
		} else {
			maxIter = 100
		}
	}
	if numThreads < 1 {
		numThreads = 1
	}

	// Ensure AdaGrad state is allocated (important after LoadModelState/LoadVectors)
	g.InitOptimizerState()

	baseLR := LEARNING_RATE
	startTime := time.Now()

	for iter := 0; iter < maxIter; iter++ {
		// Simple linear annealing with a small floor
		iterLR := baseLR
		if maxIter > 1 {
			iterLR = baseLR * (1.0 - float64(iter)/float64(maxIter))
			if iterLR < baseLR*0.1 {
				iterLR = baseLR * 0.1
			}
		}

		// Shuffle co-occurrence entries for SGD
		rand.Shuffle(len(g.Cooccur), func(i, j int) {
			g.Cooccur[i], g.Cooccur[j] = g.Cooccur[j], g.Cooccur[i]
		})

		totalCost := 0.0
		block := (len(g.Cooccur) + numThreads - 1) / numThreads

		var wg sync.WaitGroup
		costCh := make(chan float64, numThreads)

		for t := 0; t < numThreads; t++ {
			start := t * block
			end := start + block
			if start >= len(g.Cooccur) {
				break
			}
			if end > len(g.Cooccur) {
				end = len(g.Cooccur)
			}

			wg.Add(1)
			go func(start, end int, lr float64) {
				defer wg.Done()
				localCost := 0.0

				// Reusable per-goroutine buffers to avoid allocations in the inner loop
				wValues := make([]float64, g.VectorSize)
				wTildeValues := make([]float64, g.VectorSize)

				for idx := start; idx < end; idx++ {
					entry := g.Cooccur[idx]
					i := entry.Word1
					j := entry.Word2
					xij := entry.Count

					var bValue, bTildeValue float64

					// Copy parameters under shard locks into local buffers
					g.withShardLocks(i, j, func() {
						copy(wValues, g.W[i])
						copy(wTildeValues, g.WTilde[j])
						bValue = g.B[i]
						bTildeValue = g.BTilde[j]
					})

					// Dot product outside the lock
					dot := 0.0
					for k := 0; k < g.VectorSize; k++ {
						dot += wValues[k] * wTildeValues[k]
					}

					diff := dot + bValue + bTildeValue - math.Log(xij)
					weight := weightingFunction(xij)

					localCost += weight * diff * diff
					scale := 2.0 * weight * diff

					// Apply AdaGrad updates under shard locks
					g.withShardLocks(i, j, func() {
						g.updateParameters(i, j, scale, lr)
					})
				}

				costCh <- localCost
			}(start, end, iterLR)
		}

		wg.Wait()
		close(costCh)
		for c := range costCh {
			totalCost += c
		}

		if callback != nil {
			progress := TrainingProgress{
				Iteration:     iter + 1,
				MaxIterations: maxIter,
				Cost:          totalCost,
				TimeElapsed:   time.Since(startTime),
			}
			callback(progress)
		}
	}
}

// GetWordVector returns the final vector for a word (W + W̃)
func (g *GloVe) GetWordVector(word string) ([]float64, bool) {
	idx, ok := g.Vocab[word]
	if !ok {
		return nil, false
	}

	// According to the paper, use the sum W + W̃
	vector := make([]float64, g.VectorSize)
	for i := 0; i < g.VectorSize; i++ {
		vector[i] = g.W[idx][i] + g.WTilde[idx][i]
	}

	return vector, true
}

// CosineSimilarity computes cosine similarity between two vectors
func CosineSimilarity(v1, v2 []float64) float64 {
	if len(v1) != len(v2) {
		return 0
	}

	dot := 0.0
	norm1 := 0.0
	norm2 := 0.0

	for i := range v1 {
		dot += v1[i] * v2[i]
		norm1 += v1[i] * v1[i]
		norm2 += v2[i] * v2[i]
	}

	if norm1 == 0 || norm2 == 0 {
		return 0
	}

	return dot / (math.Sqrt(norm1) * math.Sqrt(norm2))
}

// WordAnalogy solves analogy tasks: a:b :: c:?
func (g *GloVe) WordAnalogy(a, b, c string, topN int) []string {
	vecA, okA := g.GetWordVector(a)
	vecB, okB := g.GetWordVector(b)
	vecC, okC := g.GetWordVector(c)

	if !okA || !okB || !okC {
		return nil
	}

	// Compute vector for the answer: b - a + c
	target := make([]float64, g.VectorSize)
	for i := 0; i < g.VectorSize; i++ {
		target[i] = vecB[i] - vecA[i] + vecC[i]
	}

	// Find nearest words
	type WordSim struct {
		Word string
		Sim  float64
	}

	var similarities []WordSim
	excludeWords := map[string]bool{a: true, b: true, c: true}

	for word := range g.Vocab {
		if excludeWords[word] {
			continue
		}

		vec, _ := g.GetWordVector(word)
		sim := CosineSimilarity(target, vec)
		similarities = append(similarities, WordSim{word, sim})
	}

	// Sort by descending similarity
	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].Sim > similarities[j].Sim
	})

	// Return top-N words
	result := make([]string, 0, topN)
	for i := 0; i < topN && i < len(similarities); i++ {
		result = append(result, similarities[i].Word)
	}

	return result
}
