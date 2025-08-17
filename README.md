# go-glove

A Go native implementation of the GloVe (Global Vectors for Word Representation) algorithm for training word embeddings from text corpora.

## What is GloVe?

GloVe is an unsupervised learning algorithm for obtaining vector representations of words. It combines the advantages of two major model families in the literature: global matrix factorization and local context window methods. The algorithm factorizes the logarithm of the word co-occurrence matrix to produce word vectors that capture semantic relationships between words.

Key features:
- Captures both global statistics and local context
- Produces meaningful word analogies (e.g., king - man + woman = queen)
- Fast training on large corpora
- Linear structure in vector space reflects semantic relationships

## Installation

```bash
go get github.com/n0madic/go-glove
```

Or clone the repository:
```bash
git clone https://github.com/n0madic/go-glove.git
cd go-glove
```

## Quick Start

### Training Word Vectors

```go
package main

import (
    "fmt"
    "os"
    "strings"
    "github.com/n0madic/go-glove"
)

func main() {
    // Create a new GloVe model
    model := glove.NewGloVe()

    // Option 1: Read and tokenize corpus
    file, err := os.Open("corpus.txt")
    if err != nil {
        panic(err)
    }
    defer file.Close()

    // Tokenize text with GloVe-optimized settings
    wordFreqs := glove.Tokenize(file, 5) // min frequency = 5

    // Convert to tokens (expand by frequency)
    var tokens []string
    for _, wf := range wordFreqs {
        for i := 0; i < wf.Freq; i++ {
            tokens = append(tokens, wf.Word)
        }
    }

    // Option 2: Use pre-tokenized text
    // tokens := strings.Fields("hello world hello world ...")

    // Build vocabulary from tokens
    err = model.BuildVocab(tokens)
    if err != nil {
        panic(err)
    }

    // Build co-occurrence matrix
    err = model.BuildCooccurrenceMatrix(tokens, 15) // window size = 15
    if err != nil {
        panic(err)
    }

    // Initialize parameters
    model.InitializeParameters(100) // vector size = 100

    // Train the model
    model.Train(50, 8) // 50 iterations, 8 threads

    // Save vectors
    model.SaveVectors("vectors.txt")
}
```

### Using Pre-trained Vectors

```go
// Load pre-trained vectors
model := glove.NewGloVe()
err := model.LoadVectors("vectors.txt")
if err != nil {
    panic(err)
}

// Get word vector
vector, exists := model.GetWordVector("king")
if exists {
    fmt.Printf("Vector for 'king': %v\n", vector)
}

// Perform word analogy: king - man + woman = ?
results := model.WordAnalogy("king", "man", "woman", 5)
for i, word := range results {
    fmt.Printf("%d. %s\n", i+1, word)
}
```

## Command Line Usage

### Training Examples

Build and run the training utilities:

```bash
# Build all utilities
go build -o train ./train
go build -o evaluate ./evaluate
go build -o tokenize ./tokenize

# Train with pre-tokenized corpus (Stanford GloVe format)
./train -corpus tokenized.txt -vector-size 200 -iterations 50

# Train with raw text and automatic tokenization
./train -corpus raw_text.txt -tokenize -vector-size 200 -iterations 50

# Tokenize text separately (recommended for large corpora)
./tokenize -input raw_text.txt -output tokenized.txt
./train -corpus tokenized.txt -vector-size 300

# Save in binary format
./train -corpus data.txt -output-format binary

# Save only word vectors (smaller file)
./train -corpus big_corpus.txt -save-mode word-only -output my_vectors.txt
```

### Training Options

The training utility supports many command-line flags:

```bash
./train -help
```

Key parameters:
- `-corpus`: Path to text corpus file (default: "corpus.txt")
- `-tokenize`: Apply GloVe tokenization to raw text (default: false, expects pre-tokenized)
- `-vector-size`: Vector dimensionality (default: 300)
- `-window-size`: Context window size (default: 10)
- `-iterations`: Number of training iterations (default: 100)
- `-threads`: Number of parallel threads (default: CPU count)
- `-output`: Output file for vectors (default: "glove_vectors.txt")
- `-save-mode`: Vector save mode (default: "word-context")
  - `all-params`: Save all parameters including biases (W + W̃ + biases)
  - `word-only`: Save word vectors only (W)
  - `word-context`: Save word + context vectors (W + W̃) - Stanford default
  - `separate`: Save word and context vectors separately (W concatenated with W̃)
- `-output-format`: Output format (default: "text")
  - `text`: Text format
  - `binary`: Binary format (Stanford GloVe compatible)
  - `both`: Both text and binary formats
- `-save-header`: Include header (vocab_size vector_size) in output file
- `-examples`: Run example analogies after training
- `-save-state`: Save model state for later continuation
- `-load-state`: Load and continue training from saved state

### Text Tokenization

Use the tokenization utility to preprocess text corpora with GloVe-optimized settings:

```bash
# Build the tokenize binary
go build -o tokenize ./tokenize

# Basic tokenization (GloVe optimized: lowercase, digit normalization, etc.)
./tokenize -input raw_text.txt -output tokenized.txt

# Tokenize with custom options
./tokenize -input corpus.txt -output tokens.txt -lowercase=false -digits-to-zero=false

# Use in pipeline (stdin/stdout)
cat large_corpus.txt | ./tokenize -input - -output - > tokenized.txt

# Generate word frequency list instead of tokens
./tokenize -input corpus.txt -show-freqs -min-freq 10 > vocab.txt

# Show help for all options
./tokenize -help
```

#### Tokenization Options

- `-input`: Input text file (use "-" for stdin)
- `-output`: Output file (use "-" for stdout, default: stdout)
- `-lowercase`: Convert to lowercase (default: true)
- `-digits-to-zero`: Normalize digits to 0 (default: true)
- `-replace-urls`: Replace URLs with `<url>` token (default: true)
- `-replace-emails`: Replace emails with `<email>` token (default: true)
- `-keep-hyphens`: Keep hyphenated words intact (default: true)
- `-min-freq`: Minimum word frequency (default: 1)
- `-show-freqs`: Output word frequencies instead of tokens (default: false)

The tokenizer applies GloVe-optimized preprocessing including:
- PTB-style contraction splitting (e.g., "can't" → "ca n't")
- Unicode normalization for quotes and dashes
- URL and email replacement with special tokens
- Digit normalization for better vocabulary compactness
- Hyphen preservation for compound words

### Evaluating Trained Models

Use the evaluate utility to test word analogies and find similar words:

```bash
# Build the evaluate binary
go build -o evaluate ./evaluate

# Basic evaluation
./evaluate -vectors glove_vectors.txt

# Custom similarity word and analogy
./evaluate -vectors my_vectors.txt -similarity-word "technology" -analogy "paris:france::london"

# Show more results
./evaluate -top-similar 15 -top-n 10
```

### Evaluation Options

- `-vectors`: Path to trained vectors file (default: "glove_vectors.txt")
- `-similarity-word`: Word to find similarities for (default: "computer")
- `-analogy`: Analogy in format "a:b::c" (default: "king:queen::man")
- `-top-similar`: Number of similar words to show (default: 10)
- `-top-n`: Number of analogy results to show (default: 5)

## API Reference

### Core Methods

#### Training Pipeline
```go
// Token-based API (recommended)
tokens := []string{"hello", "world", "hello", "..."} // pre-tokenized text
model.BuildVocab(tokens []string) error
model.BuildCooccurrenceMatrix(tokens []string, windowSize int) error
model.InitializeParameters(vectorSize int)
model.Train(maxIter, numThreads int)

// Tokenization (for preprocessing)
import "strings"
reader := strings.NewReader("Hello world! How are you?")
wordFreqs := glove.Tokenize(reader, minFreq) // Returns []glove.WordFreq
```

#### Vector Operations
```go
vector, exists := model.GetWordVector(word string) ([]float64, bool)
results := model.WordAnalogy(a, b, c string, topN int) []string
model.SaveVectors(filename string) error
model.LoadVectors(filename string) error
```

#### Advanced Saving/Loading
```go
// Save with different modes (Stanford GloVe compatible)
model.SaveVectorsMode(filename string, mode SaveMode, format OutputFormat, header bool) error

// Save/load complete model state
model.SaveModelState(filename string, includeGrads, includeCooccur bool) error
model.LoadModelState(filename string) error
```

### Configuration

Default parameters (defined in `glove.go`):
- `XMAX = 100.0` - Co-occurrence weighting cutoff
- `ALPHA = 0.75` - Weighting function exponent
- `LEARNING_RATE = 0.05` - Initial learning rate
- `MIN_COUNT = 5` - Minimum word frequency threshold

## File Formats

### Vector Files
Text format with space-separated values:
```
word1 0.1234 -0.5678 0.9012 ...
word2 0.2468 -0.1357 0.8024 ...
```

### Binary Format
Compatible with Stanford GloVe binary format for interoperability.

## Performance

- **Memory efficient**: Sparse co-occurrence matrix representation
- **Parallel training**: Configurable thread count
- **Large corpora**: Handles multi-gigabyte text files
- **No dependencies**: Uses only Go standard library
- **Flexible preprocessing**: Separate tokenization allows corpus reuse and custom pipelines
- **Stanford compatibility**: Can use existing tokenized corpora from Stanford GloVe

## Examples

See `train/main.go` for a complete example including:
- Corpus preprocessing
- Model training
- Vector evaluation
- Word analogy demonstrations

## Stanford GloVe Compatibility

This implementation is compatible with the original Stanford GloVe:
- Same algorithm and mathematical formulation
- Compatible binary vector format
- Multiple save modes matching Stanford output options

## References

- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) (Original paper)
- [Stanford GloVe Project](https://github.com/stanfordnlp/GloVe)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite the original GloVe paper:

```
@inproceedings{pennington2014glove,
  title={Glove: Global Vectors for Word Representation},
  author={Jeffrey Pennington and Richard Socher and Christopher Manning},
  booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
  year={2014}
}
```
