package glove

import (
	"bufio"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// ModelState contains the complete state of a GloVe model for serialization
type ModelState struct {
	// Model parameters
	VectorSize   int
	Symmetric    bool
	MaxVocabSize int

	// Vocabulary data
	Vocab     map[string]int
	InvVocab  []string
	WordCount []int
	VocabSize int

	// Weight matrices (always included)
	W      [][]float64
	WTilde [][]float64
	B      []float64
	BTilde []float64

	// Optional: AdaGrad gradients for continuing training
	IncludeGradients bool
	GradSqW          [][]float64
	GradSqWTilde     [][]float64
	GradSqB          []float64
	GradSqBTilde     []float64

	// Optional: Co-occurrence matrix (can be very large)
	IncludeCooccur bool
	Cooccur        []CoocEntry
}

// SaveMode defines the different modes for saving vectors compatible with Stanford GloVe
type SaveMode int

const (
	// SaveAllParams saves all parameters including biases (W + W̃ + biases)
	SaveAllParams SaveMode = iota
	// SaveWordOnly saves word vectors only (W)
	SaveWordOnly
	// SaveWordAndContext saves word + context vectors (W + W̃) - default Stanford format
	SaveWordAndContext
	// SaveSeparateVectors saves word and context vectors separately (W concatenated with W̃)
	SaveSeparateVectors
)

// OutputFormat defines the output format for saving vectors
type OutputFormat int

const (
	// OutputText saves vectors in text format (default)
	OutputText OutputFormat = iota
	// OutputBinary saves vectors in binary format
	OutputBinary
	// OutputBoth saves vectors in both text and binary formats
	OutputBoth
)

// SaveModelState saves the complete model state to a file using gob encoding
// includeGrads: whether to include AdaGrad gradients (needed for continuing training)
// includeCooccur: whether to include co-occurrence matrix (can be very large)
func (g *GloVe) SaveModelState(filename string, includeGrads, includeCooccur bool) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Create ModelState
	state := ModelState{
		// Model parameters
		VectorSize:   g.VectorSize,
		Symmetric:    g.Symmetric,
		MaxVocabSize: g.MaxVocabSize,

		// Vocabulary data
		Vocab:     g.Vocab,
		InvVocab:  g.InvVocab,
		WordCount: g.WordCount,
		VocabSize: g.VocabSize,

		// Weight matrices
		W:      g.W,
		WTilde: g.WTilde,
		B:      g.B,
		BTilde: g.BTilde,

		// Flags
		IncludeGradients: includeGrads,
		IncludeCooccur:   includeCooccur,
	}

	// Include gradients if requested
	if includeGrads && g.GradSqW != nil {
		state.GradSqW = g.GradSqW
		state.GradSqWTilde = g.GradSqWTilde
		state.GradSqB = g.GradSqB
		state.GradSqBTilde = g.GradSqBTilde
	}

	// Include co-occurrence matrix if requested
	if includeCooccur && g.Cooccur != nil {
		state.Cooccur = g.Cooccur
	}

	// Encode using gob
	encoder := gob.NewEncoder(file)
	return encoder.Encode(state)
}

// LoadModelState loads the complete model state from a file using gob decoding
func (g *GloVe) LoadModelState(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Decode using gob
	var state ModelState
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&state); err != nil {
		return err
	}

	// Restore model parameters
	g.VectorSize = state.VectorSize
	g.Symmetric = state.Symmetric
	g.MaxVocabSize = state.MaxVocabSize

	// Restore vocabulary data
	g.Vocab = state.Vocab
	g.InvVocab = state.InvVocab
	g.WordCount = state.WordCount
	g.VocabSize = state.VocabSize

	// Restore weight matrices
	g.W = state.W
	g.WTilde = state.WTilde
	g.B = state.B
	g.BTilde = state.BTilde

	// Restore gradients if included
	if state.IncludeGradients {
		g.GradSqW = state.GradSqW
		g.GradSqWTilde = state.GradSqWTilde
		g.GradSqB = state.GradSqB
		g.GradSqBTilde = state.GradSqBTilde
	} else {
		// Initialize optimizer state to prevent panic during training
		g.InitOptimizerState()
	}

	// Restore co-occurrence matrix if included
	if state.IncludeCooccur {
		g.Cooccur = state.Cooccur
	} else {
		g.Cooccur = nil
	}

	return nil
}

// SaveVectorsMode saves vectors in different formats compatible with Stanford GloVe
// mode: SaveModeAllParams, SaveModeWordOnly, SaveModeWordAndContext (default), SaveModeSeparateVectors
// format: OutputText (default), OutputBinary, OutputBoth
// header: true to include vocab_size and vector_size header line
func (g *GloVe) SaveVectorsMode(filename string, mode SaveMode, format OutputFormat, header bool) error {
	if format == OutputBinary || format == OutputBoth {
		// Save binary format
		if err := g.saveBinaryVectors(filename+".bin", mode); err != nil {
			return err
		}
	}

	if format == OutputText || format == OutputBoth {
		// Save text format
		return g.saveTextVectors(filename, mode, header)
	}

	return nil
}

// saveTextVectors saves vectors in text format
func (g *GloVe) saveTextVectors(filename string, mode SaveMode, header bool) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)

	// Write header if requested
	if header {
		switch mode {
		case SaveAllParams, SaveWordOnly, SaveWordAndContext:
			fmt.Fprintf(writer, "%d %d\n", g.VocabSize, g.VectorSize)
		case SaveSeparateVectors:
			// For mode 3, vectors are concatenated (W followed by W̃)
			fmt.Fprintf(writer, "%d %d\n", g.VocabSize, g.VectorSize*2)
		}
	}

	// Write vectors according to mode
	for i := 0; i < g.VocabSize; i++ {
		fmt.Fprintf(writer, "%s", g.InvVocab[i])

		switch mode {
		case SaveAllParams: // All parameters including biases
			for j := 0; j < g.VectorSize; j++ {
				value := g.W[i][j] + g.WTilde[i][j]
				fmt.Fprintf(writer, " %.6f", value)
			}
			// Add biases
			fmt.Fprintf(writer, " %.6f", g.B[i]+g.BTilde[i])

		case SaveWordOnly: // Word vectors only
			for j := 0; j < g.VectorSize; j++ {
				fmt.Fprintf(writer, " %.6f", g.W[i][j])
			}

		case SaveWordAndContext: // Word + context vectors (default, current behavior)
			for j := 0; j < g.VectorSize; j++ {
				value := g.W[i][j] + g.WTilde[i][j]
				fmt.Fprintf(writer, " %.6f", value)
			}

		case SaveSeparateVectors: // Word and context vectors separately (concatenated)
			// First write word vector
			for j := 0; j < g.VectorSize; j++ {
				fmt.Fprintf(writer, " %.6f", g.W[i][j])
			}
			// Then write context vector
			for j := 0; j < g.VectorSize; j++ {
				fmt.Fprintf(writer, " %.6f", g.WTilde[i][j])
			}
		}

		fmt.Fprintln(writer)
	}

	return writer.Flush()
}

// saveBinaryVectors saves vectors in binary format compatible with Stanford GloVe
func (g *GloVe) saveBinaryVectors(filename string, mode SaveMode) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Stanford GloVe binary format uses little-endian double precision floats
	// Each value is written as 8 bytes (float64)

	switch mode {
	case SaveAllParams:
		// All parameters including biases (W + W̃ + biases)
		// Format: 2 * vocab_size * (vector_size + 1) values
		// Layout: [w1_vec1, w1_vec2, ..., w1_vecN, w1_bias, w1_ctx_vec1, ..., w1_ctx_vecN, w1_ctx_bias, ...]
		for idx := 0; idx < g.VocabSize; idx++ {
			// Write word vector W[idx] with bias
			for i := 0; i < g.VectorSize; i++ {
				if err := binary.Write(file, binary.LittleEndian, g.W[idx][i]); err != nil {
					return err
				}
			}
			// Write word bias
			if err := binary.Write(file, binary.LittleEndian, g.B[idx]); err != nil {
				return err
			}

			// Write context vector W̃[idx] with bias
			for i := 0; i < g.VectorSize; i++ {
				if err := binary.Write(file, binary.LittleEndian, g.WTilde[idx][i]); err != nil {
					return err
				}
			}
			// Write context bias
			if err := binary.Write(file, binary.LittleEndian, g.BTilde[idx]); err != nil {
				return err
			}
		}

	case SaveWordOnly:
		// Word vectors only (W)
		// Format: vocab_size * vector_size values
		for idx := 0; idx < g.VocabSize; idx++ {
			for i := 0; i < g.VectorSize; i++ {
				if err := binary.Write(file, binary.LittleEndian, g.W[idx][i]); err != nil {
					return err
				}
			}
		}

	case SaveWordAndContext:
		// Word + context vectors (W + W̃) - default Stanford format
		// Format: vocab_size * vector_size * 2 values
		// Layout: [w1_vec1, w1_vec2, ..., w1_vecN, w1_ctx_vec1, ..., w1_ctx_vecN, ...]
		for idx := 0; idx < g.VocabSize; idx++ {
			// Write word vector W[idx]
			for i := 0; i < g.VectorSize; i++ {
				if err := binary.Write(file, binary.LittleEndian, g.W[idx][i]); err != nil {
					return err
				}
			}

			// Write context vector W̃[idx]
			for i := 0; i < g.VectorSize; i++ {
				if err := binary.Write(file, binary.LittleEndian, g.WTilde[idx][i]); err != nil {
					return err
				}
			}
		}

	case SaveSeparateVectors:
		// Word and context vectors separately (W concatenated with W̃)
		// Format: 2 * vocab_size * vector_size values
		// Layout: [all word vectors, then all context vectors]

		// First write all word vectors
		for idx := 0; idx < g.VocabSize; idx++ {
			for i := 0; i < g.VectorSize; i++ {
				if err := binary.Write(file, binary.LittleEndian, g.W[idx][i]); err != nil {
					return err
				}
			}
		}

		// Then write all context vectors
		for idx := 0; idx < g.VocabSize; idx++ {
			for i := 0; i < g.VectorSize; i++ {
				if err := binary.Write(file, binary.LittleEndian, g.WTilde[idx][i]); err != nil {
					return err
				}
			}
		}

	default:
		return fmt.Errorf("invalid mode %v for binary format", mode)
	}

	return nil
}

// SaveVectors saves vectors in default format (word+context mode, text format, no header)
func (g *GloVe) SaveVectors(filename string) error {
	return g.SaveVectorsMode(filename, SaveWordAndContext, OutputText, false)
}

// LoadVectors loads vectors from a file with auto-detection of format
func (g *GloVe) LoadVectors(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	// Read all lines for analysis
	var lines []string
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading file: %v", err)
	}

	if len(lines) == 0 {
		return fmt.Errorf("empty file")
	}

	// Check if first line is a header (vocab_size vector_size)
	headerSkip := 0
	firstLine := lines[0]
	if parts := strings.Fields(firstLine); len(parts) == 2 {
		// Try to parse as vocab_size vector_size
		if vocabSize, err1 := strconv.Atoi(parts[0]); err1 == nil {
			if vectorSize, err2 := strconv.Atoi(parts[1]); err2 == nil {
				// Valid header detected
				if vocabSize > 0 && vectorSize > 0 && vocabSize == len(lines)-1 {
					headerSkip = 1
					g.VocabSize = vocabSize
					g.VectorSize = vectorSize
				}
			}
		}
	}

	// If no header detected, analyze first data line
	if headerSkip == 0 {
		g.VocabSize = len(lines)
		dataLine := lines[0]

		spaceIdx := strings.Index(dataLine, " ")
		if spaceIdx == -1 {
			return fmt.Errorf("invalid format: line should contain word and vector components separated by space")
		}

		vectorStr := dataLine[spaceIdx+1:]
		parts := strings.Fields(vectorStr)
		if len(parts) < 1 {
			return fmt.Errorf("invalid format: line should contain at least one vector component")
		}

		g.VectorSize = len(parts)
	}

	// Initialize data structures
	g.Vocab = make(map[string]int, g.VocabSize)
	g.InvVocab = make([]string, g.VocabSize)
	g.W = make([][]float64, g.VocabSize)
	g.WTilde = make([][]float64, g.VocabSize)

	// Process data lines
	for i := headerSkip; i < len(lines); i++ {
		line := lines[i]
		lineIdx := i - headerSkip

		// Split line into word and vector components
		spaceIdx := strings.Index(line, " ")
		if spaceIdx == -1 {
			return fmt.Errorf("invalid format at line %d: line should contain word and vector components separated by space", i+1)
		}

		word := line[:spaceIdx]
		vectorStr := line[spaceIdx+1:]
		parts := strings.Fields(vectorStr)

		// Check vector size consistency and detect mode
		expectedSize := g.VectorSize
		if len(parts) == g.VectorSize*2 {
			// Mode 3 detected: W and W̃ concatenated
			expectedSize = g.VectorSize * 2
		} else if len(parts) == g.VectorSize+1 {
			// Mode 0 detected: includes bias
			expectedSize = g.VectorSize + 1
		}

		if len(parts) != expectedSize {
			return fmt.Errorf("invalid format at line %d: expected %d vector components, got %d",
				i+1, expectedSize, len(parts))
		}

		g.Vocab[word] = lineIdx
		g.InvVocab[lineIdx] = word
		g.W[lineIdx] = make([]float64, g.VectorSize)
		g.WTilde[lineIdx] = make([]float64, g.VectorSize)

		// Parse vectors based on detected mode
		if len(parts) == g.VectorSize*2 {
			// Mode 3: separate W and W̃
			for j := 0; j < g.VectorSize; j++ {
				val, err := parseFloat(parts[j])
				if err != nil {
					return fmt.Errorf("error parsing W vector component for word '%s': %v", word, err)
				}
				g.W[lineIdx][j] = val
			}
			for j := 0; j < g.VectorSize; j++ {
				val, err := parseFloat(parts[j+g.VectorSize])
				if err != nil {
					return fmt.Errorf("error parsing W̃ vector component for word '%s': %v", word, err)
				}
				g.WTilde[lineIdx][j] = val
			}
		} else if len(parts) == g.VectorSize+1 {
			// Mode 0: W + W̃ + bias, split equally
			for j := 0; j < g.VectorSize; j++ {
				val, err := parseFloat(parts[j])
				if err != nil {
					return fmt.Errorf("error parsing vector component for word '%s': %v", word, err)
				}
				g.W[lineIdx][j] = val / 2.0
				g.WTilde[lineIdx][j] = val / 2.0
			}
			// TODO: Handle bias (parts[g.VectorSize])
		} else {
			// Mode 1 or 2: single vector (either W only or W + W̃)
			for j := 0; j < g.VectorSize; j++ {
				val, err := parseFloat(parts[j])
				if err != nil {
					return fmt.Errorf("error parsing vector component for word '%s': %v", word, err)
				}
				// Assume mode 2 (W + W̃), split equally
				g.W[lineIdx][j] = val / 2.0
				g.WTilde[lineIdx][j] = val / 2.0
			}
		}
	}

	// Initialize biases to zero since they're typically not saved
	g.B = make([]float64, g.VocabSize)
	g.BTilde = make([]float64, g.VocabSize)

	// Initialize word counts to 1 (unknown, but needed for some operations)
	g.WordCount = make([]int, g.VocabSize)
	for i := 0; i < g.VocabSize; i++ {
		g.WordCount[i] = 1
	}

	// Initialize optimizer state to prevent panic during training
	g.InitOptimizerState()

	return nil
}

// parseFloat is a helper function to parse float64 values
func parseFloat(s string) (float64, error) {
	// Handle potential parsing issues with different float formats
	return strconv.ParseFloat(s, 64)
}
