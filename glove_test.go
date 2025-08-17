package glove

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// Test constants
const (
	TestEpsilon = 1e-6
	TestCorpus  = "the cat sat on the mat the dog ran in the park"
	// Rich corpus with words appearing at least MIN_COUNT=5 times
	RichCorpus = "the the the the the cat cat cat cat cat sat sat sat sat sat on on on on on " +
		"mat mat mat mat mat dog dog dog dog dog ran ran ran ran ran in in in in in " +
		"park park park park park"
)

// Helper function to create temporary test files
func createTempFile(t testing.TB, content string) string {
	tmpFile, err := os.CreateTemp("", "glove_test_*.txt")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer tmpFile.Close()

	if _, err := tmpFile.WriteString(content); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}

	return tmpFile.Name()
}

// Helper function to clean up temporary files
func cleanupFile(filename string) {
	os.Remove(filename)
}

func TestNewGloVe(t *testing.T) {
	model := NewGloVe()
	if model.VectorSize != 0 {
		t.Errorf("NewGloVe() vector size = %v, want 0 (uninitialized)", model.VectorSize)
	}
	if model.Symmetric != true {
		t.Errorf("NewGloVe() symmetric = %v, want true", model.Symmetric)
	}
	if model.MaxVocabSize != 400000 {
		t.Errorf("NewGloVe() max vocab size = %v, want 400000", model.MaxVocabSize)
	}
	if model.Vocab == nil {
		t.Error("NewGloVe() vocab map is nil")
	}
}

func TestBuildVocab(t *testing.T) {
	tests := []struct {
		name          string
		corpus        string
		expectedLen   int
		shouldContain []string
	}{
		{
			name:          "Basic corpus",
			corpus:        TestCorpus,
			expectedLen:   0,          // MIN_COUNT=5 filters out all words in short test corpus
			shouldContain: []string{}, // MIN_COUNT=5 filters out all words
		},
		{
			name:          "Rich corpus",
			corpus:        RichCorpus,
			expectedLen:   9, // All words appear exactly 5 times
			shouldContain: []string{"the", "cat", "sat", "on", "mat", "dog", "ran", "in", "park"},
		},
		{
			name:          "Single word repeated",
			corpus:        "hello hello hello hello hello",
			expectedLen:   1,
			shouldContain: []string{"hello"},
		},
		{
			name:          "Empty corpus",
			corpus:        "",
			expectedLen:   0,
			shouldContain: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := NewGloVe()
			filename := createTempFile(t, tt.corpus)
			defer cleanupFile(filename)

			err := model.BuildVocab(filename)
			if err != nil {
				t.Fatalf("BuildVocab() error = %v", err)
			}

			if model.VocabSize != tt.expectedLen {
				t.Errorf("BuildVocab() vocab size = %v, want %v", model.VocabSize, tt.expectedLen)
			}

			// Check that expected words are in vocabulary
			for _, word := range tt.shouldContain {
				if _, exists := model.Vocab[word]; !exists {
					t.Errorf("BuildVocab() missing word '%s' in vocabulary", word)
				}
			}

			// Check consistency between Vocab and InvVocab
			if len(model.Vocab) != len(model.InvVocab) {
				t.Errorf("BuildVocab() vocab and inv_vocab size mismatch: %d vs %d",
					len(model.Vocab), len(model.InvVocab))
			}

			for word, idx := range model.Vocab {
				if model.InvVocab[idx] != word {
					t.Errorf("BuildVocab() vocab/inv_vocab inconsistency: %s at index %d maps to %s",
						word, idx, model.InvVocab[idx])
				}
			}
		})
	}
}

func TestBuildCooccurrenceMatrix(t *testing.T) {
	model := NewGloVe()
	filename := createTempFile(t, RichCorpus)
	defer cleanupFile(filename)

	// First build vocabulary
	err := model.BuildVocab(filename)
	if err != nil {
		t.Fatalf("BuildVocab() error = %v", err)
	}

	// Then build co-occurrence matrix
	err = model.BuildCooccurrenceMatrix(filename, 2)
	if err != nil {
		t.Fatalf("BuildCooccurrenceMatrix() error = %v", err)
	}

	if len(model.Cooccur) == 0 {
		t.Error("BuildCooccurrenceMatrix() produced empty co-occurrence matrix")
	}

	// Check that all indices in co-occurrence matrix are valid
	for i, entry := range model.Cooccur {
		if entry.Word1 < 0 || entry.Word1 >= model.VocabSize {
			t.Errorf("BuildCooccurrenceMatrix() invalid Word1 index %d at entry %d", entry.Word1, i)
		}
		if entry.Word2 < 0 || entry.Word2 >= model.VocabSize {
			t.Errorf("BuildCooccurrenceMatrix() invalid Word2 index %d at entry %d", entry.Word2, i)
		}
		if entry.Count <= 0 {
			t.Errorf("BuildCooccurrenceMatrix() non-positive count %f at entry %d", entry.Count, i)
		}
	}

	// Test symmetric co-occurrence
	model.Symmetric = true
	err = model.BuildCooccurrenceMatrix(filename, 1)
	if err != nil {
		t.Fatalf("BuildCooccurrenceMatrix() symmetric error = %v", err)
	}

	// Find if symmetric pairs exist
	coocMap := make(map[string]float64)
	for _, entry := range model.Cooccur {
		key := fmt.Sprintf("%d-%d", entry.Word1, entry.Word2)
		coocMap[key] = entry.Count
	}

	symmetricPairs := 0
	for _, entry := range model.Cooccur {
		reverseKey := fmt.Sprintf("%d-%d", entry.Word2, entry.Word1)
		if _, exists := coocMap[reverseKey]; exists && entry.Word1 != entry.Word2 {
			symmetricPairs++
		}
	}

	if symmetricPairs == 0 && len(model.Cooccur) > 1 {
		t.Error("BuildCooccurrenceMatrix() symmetric mode didn't produce symmetric pairs")
	}
}

func TestInitializeParameters(t *testing.T) {
	tests := []struct {
		name       string
		vectorSize int
		vocabSize  int
	}{
		{"Small model", 50, 100},
		{"Medium model", 100, 1000},
		{"Large model", 300, 5000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := NewGloVe()
			model.VocabSize = tt.vocabSize

			model.InitializeParameters(tt.vectorSize)

			// Check matrix dimensions
			if len(model.W) != tt.vocabSize {
				t.Errorf("InitializeParameters() W matrix rows = %d, want %d", len(model.W), tt.vocabSize)
			}
			if len(model.WTilde) != tt.vocabSize {
				t.Errorf("InitializeParameters() WTilde matrix rows = %d, want %d", len(model.WTilde), tt.vocabSize)
			}

			// Check vector dimensions
			for i := 0; i < tt.vocabSize; i++ {
				if len(model.W[i]) != tt.vectorSize {
					t.Errorf("InitializeParameters() W[%d] length = %d, want %d", i, len(model.W[i]), tt.vectorSize)
				}
				if len(model.WTilde[i]) != tt.vectorSize {
					t.Errorf("InitializeParameters() WTilde[%d] length = %d, want %d", i, len(model.WTilde[i]), tt.vectorSize)
				}
			}

			// Check bias dimensions
			if len(model.B) != tt.vocabSize {
				t.Errorf("InitializeParameters() B length = %d, want %d", len(model.B), tt.vocabSize)
			}
			if len(model.BTilde) != tt.vocabSize {
				t.Errorf("InitializeParameters() BTilde length = %d, want %d", len(model.BTilde), tt.vocabSize)
			}

			// Check gradient dimensions
			if len(model.GradSqW) != tt.vocabSize {
				t.Errorf("InitializeParameters() GradSqW length = %d, want %d", len(model.GradSqW), tt.vocabSize)
			}

			// Check that parameters are within expected range
			initRange := 0.5 / float64(tt.vectorSize)
			for i := 0; i < tt.vocabSize; i++ {
				for j := 0; j < tt.vectorSize; j++ {
					if math.Abs(model.W[i][j]) > initRange {
						t.Errorf("InitializeParameters() W[%d][%d] = %f exceeds init range %f",
							i, j, model.W[i][j], initRange)
					}
					if model.GradSqW[i][j] != 1.0 {
						t.Errorf("InitializeParameters() GradSqW[%d][%d] = %f, want 1.0",
							i, j, model.GradSqW[i][j])
					}
				}
				if model.GradSqB[i] != 1.0 {
					t.Errorf("InitializeParameters() GradSqB[%d] = %f, want 1.0", i, model.GradSqB[i])
				}
			}
		})
	}
}

func TestWeightingFunction(t *testing.T) {
	tests := []struct {
		name string
		x    float64
		want float64
	}{
		{"Below XMAX", 50.0, math.Pow(50.0/XMAX, ALPHA)},
		{"At XMAX", XMAX, 1.0},
		{"Above XMAX", 200.0, 1.0},
		{"Zero", 0.0, 0.0},
		{"Small value", 1.0, math.Pow(1.0/XMAX, ALPHA)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := weightingFunction(tt.x)
			if math.Abs(got-tt.want) > TestEpsilon {
				t.Errorf("weightingFunction(%f) = %f, want %f", tt.x, got, tt.want)
			}
		})
	}
}

func TestGetWordVector(t *testing.T) {
	model := NewGloVe()
	model.VocabSize = 2
	model.VectorSize = 3
	model.Vocab = map[string]int{"hello": 0, "world": 1}
	model.InvVocab = []string{"hello", "world"}

	// Initialize with known values
	model.W = [][]float64{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}
	model.WTilde = [][]float64{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}

	tests := []struct {
		name     string
		word     string
		expected []float64
		found    bool
	}{
		{"Existing word hello", "hello", []float64{1.1, 2.2, 3.3}, true},
		{"Existing word world", "world", []float64{4.4, 5.5, 6.6}, true},
		{"Non-existing word", "nonexistent", nil, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vector, found := model.GetWordVector(tt.word)

			if found != tt.found {
				t.Errorf("GetWordVector(%s) found = %v, want %v", tt.word, found, tt.found)
			}

			if tt.found {
				if len(vector) != len(tt.expected) {
					t.Errorf("GetWordVector(%s) vector length = %d, want %d",
						tt.word, len(vector), len(tt.expected))
				}

				for i, val := range vector {
					if math.Abs(val-tt.expected[i]) > TestEpsilon {
						t.Errorf("GetWordVector(%s) vector[%d] = %f, want %f",
							tt.word, i, val, tt.expected[i])
					}
				}
			}
		})
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		v1   []float64
		v2   []float64
		want float64
	}{
		{"Identical vectors", []float64{1, 2, 3}, []float64{1, 2, 3}, 1.0},
		{"Orthogonal vectors", []float64{1, 0}, []float64{0, 1}, 0.0},
		{"Opposite vectors", []float64{1, 2}, []float64{-1, -2}, -1.0},
		{"Different lengths", []float64{1, 2}, []float64{1, 2, 3}, 0.0},
		{"Zero vector", []float64{0, 0}, []float64{1, 2}, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CosineSimilarity(tt.v1, tt.v2)
			if math.Abs(got-tt.want) > TestEpsilon {
				t.Errorf("CosineSimilarity(%v, %v) = %f, want %f", tt.v1, tt.v2, got, tt.want)
			}
		})
	}
}

func TestWordAnalogy(t *testing.T) {
	model := NewGloVe()
	model.VocabSize = 4
	model.Vocab = map[string]int{"king": 0, "man": 1, "woman": 2, "queen": 3}
	model.InvVocab = []string{"king", "man", "woman", "queen"}

	// Set up vectors for classic analogy: king - man + woman ≈ queen
	model.W = [][]float64{
		{1.0, 0.0}, // king
		{0.5, 0.0}, // man
		{0.5, 1.0}, // woman
		{1.0, 1.0}, // queen
	}
	model.WTilde = [][]float64{
		{0.0, 0.0}, // king
		{0.0, 0.0}, // man
		{0.0, 0.0}, // woman
		{0.0, 0.0}, // queen
	}

	tests := []struct {
		name     string
		a, b, c  string
		topN     int
		expected string
	}{
		{"Classic analogy", "king", "man", "woman", 1, "queen"},
		{"Non-existent word", "king", "nonexistent", "woman", 1, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := model.WordAnalogy(tt.a, tt.b, tt.c, tt.topN)

			if tt.expected == "" {
				if len(result) != 0 {
					t.Errorf("WordAnalogy(%s, %s, %s, %d) = %v, want empty",
						tt.a, tt.b, tt.c, tt.topN, result)
				}
			} else {
				if len(result) == 0 || result[0] != tt.expected {
					t.Errorf("WordAnalogy(%s, %s, %s, %d) = %v, want [%s]",
						tt.a, tt.b, tt.c, tt.topN, result, tt.expected)
				}
			}
		})
	}
}

func TestTrain(t *testing.T) {
	// Create a small test model
	model := NewGloVe()
	filename := createTempFile(t, RichCorpus)
	defer cleanupFile(filename)

	// Build vocab and co-occurrence matrix
	err := model.BuildVocab(filename)
	if err != nil {
		t.Fatalf("BuildVocab() error = %v", err)
	}

	err = model.BuildCooccurrenceMatrix(filename, 2)
	if err != nil {
		t.Fatalf("BuildCooccurrenceMatrix() error = %v", err)
	}

	model.InitializeParameters(2)

	// Store initial parameters for comparison
	initialW := make([][]float64, len(model.W))
	for i := range model.W {
		initialW[i] = make([]float64, len(model.W[i]))
		copy(initialW[i], model.W[i])
	}

	// Train for a few iterations
	model.Train(5, 1)

	// Check that parameters have changed (training occurred)
	parametersChanged := false
	for i := range model.W {
		for j := range model.W[i] {
			if math.Abs(model.W[i][j]-initialW[i][j]) > TestEpsilon {
				parametersChanged = true
				break
			}
		}
		if parametersChanged {
			break
		}
	}

	if !parametersChanged {
		t.Error("Train() did not change model parameters")
	}

	// Test multi-threaded training
	model.Train(3, 2)
	// Should complete without error
}

func TestSaveLoadVectors(t *testing.T) {
	// Create test model with proper vocabulary
	model := NewGloVe()
	model.VocabSize = 2
	model.VectorSize = 3
	model.Vocab = map[string]int{"hello": 0, "world": 1}
	model.InvVocab = []string{"hello", "world"}
	model.W = [][]float64{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}}
	model.WTilde = [][]float64{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}
	model.B = []float64{0.1, 0.2}
	model.BTilde = []float64{0.01, 0.02}

	// Test different save modes
	modes := []SaveMode{SaveAllParams, SaveWordOnly, SaveWordAndContext, SaveSeparateVectors}
	modeNames := []string{"AllParams", "WordOnly", "WordAndContext", "SeparateVectors"}
	for i, mode := range modes {
		t.Run(fmt.Sprintf("Mode_%s", modeNames[i]), func(t *testing.T) {
			filename := fmt.Sprintf("test_vectors_mode_%s.txt", modeNames[i])
			defer cleanupFile(filename)

			// Save vectors
			err := model.SaveVectorsMode(filename, mode, OutputText, false)
			if err != nil {
				t.Fatalf("SaveVectorsMode() mode %v error = %v", mode, err)
			}

			// Check file exists
			if _, err := os.Stat(filename); os.IsNotExist(err) {
				t.Fatalf("SaveVectorsMode() mode %v did not create file", mode)
			}

			// Load vectors
			newModel := NewGloVe()
			err = newModel.LoadVectors(filename)
			if err != nil {
				t.Fatalf("LoadVectors() mode %v error = %v", mode, err)
			}

			// Basic checks
			if newModel.VocabSize != model.VocabSize {
				t.Errorf("LoadVectors() mode %v vocab size = %d, want %d",
					mode, newModel.VocabSize, model.VocabSize)
			}

			// Check vocabulary consistency
			for word, idx := range model.Vocab {
				if newIdx, exists := newModel.Vocab[word]; !exists || newIdx != idx {
					t.Errorf("LoadVectors() mode %v word '%s' mapping inconsistent", mode, word)
				}
			}
		})
	}
}

func TestSaveLoadVectorsWithHeader(t *testing.T) {
	model := NewGloVe()
	model.VocabSize = 3
	model.VectorSize = 2
	model.Vocab = map[string]int{"a": 0, "b": 1, "c": 2}
	model.InvVocab = []string{"a", "b", "c"}
	model.W = [][]float64{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}
	model.WTilde = [][]float64{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}

	filename := "test_vectors_header.txt"
	defer cleanupFile(filename)

	// Save with header
	err := model.SaveVectorsMode(filename, SaveWordAndContext, OutputText, true)
	if err != nil {
		t.Fatalf("SaveVectorsMode() with header error = %v", err)
	}

	// Load and verify
	newModel := NewGloVe()
	err = newModel.LoadVectors(filename)
	if err != nil {
		t.Fatalf("LoadVectors() with header error = %v", err)
	}

	if newModel.VocabSize != model.VocabSize {
		t.Errorf("LoadVectors() with header vocab size = %d, want %d",
			newModel.VocabSize, model.VocabSize)
	}

	if newModel.VectorSize != model.VectorSize {
		t.Errorf("LoadVectors() with header vector size = %d, want %d",
			newModel.VectorSize, model.VectorSize)
	}
}

func TestSaveLoadModelState(t *testing.T) {
	// Create test model with full state
	model := NewGloVe()
	model.VocabSize = 2
	model.Vocab = map[string]int{"test": 0, "word": 1}
	model.InvVocab = []string{"test", "word"}
	model.WordCount = []int{10, 20}
	model.W = [][]float64{{1.0, 2.0}, {3.0, 4.0}}
	model.WTilde = [][]float64{{0.1, 0.2}, {0.3, 0.4}}
	model.B = []float64{0.5, 0.6}
	model.BTilde = []float64{0.05, 0.06}

	// Initialize gradients
	model.InitializeParameters(2)

	tests := []struct {
		name           string
		includeGrads   bool
		includeCooccur bool
	}{
		{"Basic state", false, false},
		{"With gradients", true, false},
		{"With co-occurrence", false, true},
		{"Complete state", true, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filename := fmt.Sprintf("test_state_%s.gob", tt.name)
			defer cleanupFile(filename)

			// Add co-occurrence data if needed
			if tt.includeCooccur {
				model.Cooccur = []CoocEntry{
					{Word1: 0, Word2: 1, Count: 5.0},
					{Word1: 1, Word2: 0, Count: 3.0},
				}
			}

			// Save model state
			err := model.SaveModelState(filename, tt.includeGrads, tt.includeCooccur)
			if err != nil {
				t.Fatalf("SaveModelState() error = %v", err)
			}

			// Load model state
			newModel := NewGloVe()
			err = newModel.LoadModelState(filename)
			if err != nil {
				t.Fatalf("LoadModelState() error = %v", err)
			}

			// Verify basic parameters
			if newModel.VocabSize != model.VocabSize {
				t.Errorf("LoadModelState() vocab size = %d, want %d",
					newModel.VocabSize, model.VocabSize)
			}

			if newModel.VectorSize != model.VectorSize {
				t.Errorf("LoadModelState() vector size = %d, want %d",
					newModel.VectorSize, model.VectorSize)
			}

			// Verify vocabulary
			for word, idx := range model.Vocab {
				if newIdx, exists := newModel.Vocab[word]; !exists || newIdx != idx {
					t.Errorf("LoadModelState() word '%s' mapping inconsistent", word)
				}
			}

			// Verify matrices
			for i := range model.W {
				for j := range model.W[i] {
					if math.Abs(newModel.W[i][j]-model.W[i][j]) > TestEpsilon {
						t.Errorf("LoadModelState() W[%d][%d] = %f, want %f",
							i, j, newModel.W[i][j], model.W[i][j])
					}
				}
			}

			// Verify gradients behavior
			if tt.includeGrads {
				if newModel.GradSqW == nil {
					t.Error("LoadModelState() gradients not loaded when includeGrads=true")
				}
			} else {
				// When includeGrads=false, InitOptimizerState() should have been called
				// to initialize gradients to prevent panic during training
				if newModel.GradSqW == nil {
					t.Error("LoadModelState() gradients not initialized when includeGrads=false")
				}
			}

			// Verify co-occurrence if included
			if tt.includeCooccur {
				if newModel.Cooccur == nil || len(newModel.Cooccur) != len(model.Cooccur) {
					t.Error("LoadModelState() co-occurrence not loaded correctly")
				}
			} else {
				if newModel.Cooccur != nil {
					t.Error("LoadModelState() co-occurrence loaded when includeCooccur=false")
				}
			}
		})
	}
}

func TestSaveLoadDefaultWrapper(t *testing.T) {
	model := NewGloVe()
	model.VocabSize = 1
	model.VectorSize = 2
	model.Vocab = map[string]int{"test": 0}
	model.InvVocab = []string{"test"}
	model.W = [][]float64{{1.0, 2.0}}
	model.WTilde = [][]float64{{0.1, 0.2}}

	filename := "test_default_save.txt"
	defer cleanupFile(filename)

	// Test SaveVectors default wrapper
	err := model.SaveVectors(filename)
	if err != nil {
		t.Fatalf("SaveVectors() error = %v", err)
	}

	// Verify it's mode 2 (W + WTilde)
	newModel := NewGloVe()
	err = newModel.LoadVectors(filename)
	if err != nil {
		t.Fatalf("LoadVectors() error = %v", err)
	}

	expectedVector, _ := model.GetWordVector("test")
	actualVector, _ := newModel.GetWordVector("test")

	for i := range expectedVector {
		if math.Abs(actualVector[i]-expectedVector[i]) > TestEpsilon {
			t.Errorf("SaveVectors() default mode vector[%d] = %f, want %f",
				i, actualVector[i], expectedVector[i])
		}
	}
}

// Benchmark tests
func BenchmarkBuildVocab(b *testing.B) {
	corpus := "the quick brown fox jumps over the lazy dog " +
		"the quick brown fox jumps over the lazy dog " +
		"the quick brown fox jumps over the lazy dog"
	filename := createTempFile(b, corpus)
	defer cleanupFile(filename)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model := NewGloVe()
		model.BuildVocab(filename)
	}
}

func BenchmarkBuildCooccurrenceMatrix(b *testing.B) {
	corpus := "the quick brown fox jumps over the lazy dog " +
		"the quick brown fox jumps over the lazy dog " +
		"the quick brown fox jumps over the lazy dog"
	filename := createTempFile(b, corpus)
	defer cleanupFile(filename)

	model := NewGloVe()
	model.BuildVocab(filename)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.BuildCooccurrenceMatrix(filename, 5)
	}
}

func BenchmarkTrain(b *testing.B) {
	model := NewGloVe()
	filename := createTempFile(b, RichCorpus+" "+RichCorpus)
	defer cleanupFile(filename)

	model.BuildVocab(filename)
	model.BuildCooccurrenceMatrix(filename, 3)
	model.InitializeParameters(100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Train(1, 1)
	}
}

func BenchmarkGetWordVector(b *testing.B) {
	model := NewGloVe()
	model.VocabSize = 1000
	model.Vocab = make(map[string]int)
	model.InvVocab = make([]string, 1000)
	model.W = make([][]float64, 1000)
	model.WTilde = make([][]float64, 1000)

	// Initialize test data
	for i := 0; i < 1000; i++ {
		word := fmt.Sprintf("word%d", i)
		model.Vocab[word] = i
		model.InvVocab[i] = word
		model.W[i] = make([]float64, 300)
		model.WTilde[i] = make([]float64, 300)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.GetWordVector("word500")
	}
}

func BenchmarkCosineSimilarity(b *testing.B) {
	v1 := make([]float64, 300)
	v2 := make([]float64, 300)
	for i := range v1 {
		v1[i] = float64(i)
		v2[i] = float64(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CosineSimilarity(v1, v2)
	}
}
