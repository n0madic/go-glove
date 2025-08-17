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
    "github.com/n0madic/go-glove"
)

func main() {
    // Create a new GloVe model
    model := glove.NewGloVe()

    // Build vocabulary from corpus
    err := model.BuildVocab("corpus.txt")
    if err != nil {
        panic(err)
    }

    // Build co-occurrence matrix
    err = model.BuildCooccurrenceMatrix("corpus.txt", 15) // window size = 15
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

### Training Example

Build and run the training example:

```bash
# Build the training binary
go build -o train ./train

# Run training with default parameters
./train

# Run training with custom parameters
./train -corpus my_corpus.txt -vector-size 200 -iterations 50 -window-size 15

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
model.BuildVocab(filename string) error
model.BuildCooccurrenceMatrix(filename string, windowSize int) error
model.InitializeParameters(vectorSize int)
model.Train(maxIter, numThreads int)
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

MIT License
