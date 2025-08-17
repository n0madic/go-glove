package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/n0madic/go-glove"
)

// parseSaveMode converts string to SaveMode enum
func parseSaveMode(mode string) (glove.SaveMode, error) {
	switch strings.ToLower(mode) {
	case "all-params", "all":
		return glove.SaveAllParams, nil
	case "word-only", "word":
		return glove.SaveWordOnly, nil
	case "word-context", "context", "default":
		return glove.SaveWordAndContext, nil
	case "separate", "sep":
		return glove.SaveSeparateVectors, nil
	default:
		return 0, fmt.Errorf("invalid save mode '%s'. Valid options: all-params, word-only, word-context, separate", mode)
	}
}

// parseOutputFormat converts string to OutputFormat enum
func parseOutputFormat(format string) (glove.OutputFormat, error) {
	switch strings.ToLower(format) {
	case "text", "txt":
		return glove.OutputText, nil
	case "binary", "bin":
		return glove.OutputBinary, nil
	case "both":
		return glove.OutputBoth, nil
	default:
		return 0, fmt.Errorf("invalid output format '%s'. Valid options: text, binary, both", format)
	}
}

func main() {
	var (
		corpusFile     = flag.String("corpus", "corpus.txt", "Path to the text corpus file (whitespace-separated tokens)")
		outputFile     = flag.String("output", "glove_vectors.txt", "Output file for trained vectors")
		vectorSize     = flag.Int("vector-size", 300, "Vector dimensionality")
		windowSize     = flag.Int("window-size", 10, "Context window size for co-occurrence matrix")
		iterations     = flag.Int("iterations", 100, "Number of training iterations")
		threads        = flag.Int("threads", runtime.NumCPU(), "Number of threads for training")
		runExamples    = flag.Bool("examples", false, "Run example usage after training (analogies and similarity)")
		similarityWord = flag.String("similarity-word", "computer", "Word to find similarities for in examples")
		analogy        = flag.String("analogy", "king:queen::man", "Analogy in format 'a:b::c' to solve for 'a:b::c:?'")
		topN           = flag.Int("top-n", 3, "Number of analogy results to show")
		topSimilar     = flag.Int("top-similar", 10, "Number of similar words to show")
		saveState      = flag.String("save-state", "", "Save model state to gob file for later continuation")
		saveInterval   = flag.Int("save-interval", 0, "How often to save state during training (0=only at end, 1=every iteration, 10=every 10th iteration, etc.)")
		loadState      = flag.String("load-state", "", "Load model state from gob file to continue training")
		quiet          = flag.Bool("quiet", false, "Disable training progress output")
		saveMode       = flag.String("save-mode", "word-context", "Vector save mode: 'all-params', 'word-only', 'word-context', 'separate'")
		outputFormat   = flag.String("output-format", "text", "Output format: 'text', 'binary', 'both'")
		saveHeader     = flag.Bool("save-header", false, "Include header (vocab_size vector_size) in output file")
		tokenize       = flag.Bool("tokenize", false, "Apply GloVe tokenization to input corpus (default: expect pre-tokenized)")
		help           = flag.Bool("help", false, "Show usage information")
	)

	flag.Parse()

	if *help {
		fmt.Printf("GloVe Training Utility\n\n")
		fmt.Printf("Usage: %s [options]\n\n", os.Args[0])
		fmt.Printf("This utility trains word embeddings using the GloVe algorithm.\n")
		fmt.Printf("By default, expects pre-tokenized corpus (whitespace-separated tokens).\n")
		fmt.Printf("Use -tokenize flag to apply GloVe tokenization to raw text.\n\n")
		fmt.Printf("Options:\n")
		flag.PrintDefaults()
		fmt.Printf("\nExamples:\n")
		fmt.Printf("  # Train on pre-tokenized corpus (Stanford GloVe format)\n")
		fmt.Printf("  %s -corpus tokenized.txt -vector-size 200 -iterations 50\n\n", os.Args[0])
		fmt.Printf("  # Train on raw text with tokenization\n")
		fmt.Printf("  %s -corpus raw_text.txt -tokenize -examples\n\n", os.Args[0])
		fmt.Printf("  # Continue training from saved state\n")
		fmt.Printf("  %s -load-state model.gob -iterations 50\n\n", os.Args[0])
		fmt.Printf("\nSave Modes:\n")
		fmt.Printf("  all-params  : Save all parameters including biases (W + W̃ + biases)\n")
		fmt.Printf("  word-only   : Save word vectors only (W)\n")
		fmt.Printf("  word-context: Save word + context vectors (W + W̃) - default Stanford format\n")
		fmt.Printf("  separate    : Save word and context vectors separately (W concatenated with W̃)\n")
		fmt.Printf("\nOutput Formats:\n")
		fmt.Printf("  text   : Text format (default)\n")
		fmt.Printf("  binary : Binary format (Stanford GloVe compatible)\n")
		fmt.Printf("  both   : Both text and binary formats\n")
		return
	}

	if *loadState == "" {
		if _, err := os.Stat(*corpusFile); os.IsNotExist(err) {
			log.Fatalf("Corpus file does not exist: %s", *corpusFile)
		}
	} else {
		if _, err := os.Stat(*loadState); os.IsNotExist(err) {
			log.Fatalf("State file does not exist: %s", *loadState)
		}
	}

	// Validate and parse the save mode and output format
	saveModeEnum, err := parseSaveMode(*saveMode)
	if err != nil {
		log.Fatalf("Invalid save mode: %v", err)
	}

	outputFormatEnum, err := parseOutputFormat(*outputFormat)
	if err != nil {
		log.Fatalf("Invalid output format: %v", err)
	}

	var model *glove.GloVe

	if *loadState != "" {
		fmt.Printf("Loading model state from: %s\n", *loadState)
		model = &glove.GloVe{}
		if err := model.LoadModelState(*loadState); err != nil {
			log.Fatalf("Failed to load model state: %v", err)
		}
		fmt.Printf("Loaded model with vocabulary size: %d\n", model.VocabSize)
		fmt.Printf("Continuing training for %d iterations\n", *iterations)
	} else {
		fmt.Printf("Training GloVe model with the following parameters:\n")
		fmt.Printf("  Corpus file: %s\n", *corpusFile)
		fmt.Printf("  Vector size: %d\n", *vectorSize)
		fmt.Printf("  Window size: %d\n", *windowSize)
		fmt.Printf("  Iterations: %d\n", *iterations)
		fmt.Printf("  Threads: %d\n", *threads)
		fmt.Printf("  Output file: %s\n", *outputFile)
		fmt.Printf("  Save mode: %s\n", *saveMode)
		fmt.Printf("  Output format: %s\n", *outputFormat)
		if *tokenize {
			fmt.Printf("  Tokenization: enabled (raw text)\n")
		} else {
			fmt.Printf("  Tokenization: disabled (pre-tokenized)\n")
		}
		if *saveHeader {
			fmt.Printf("  Include header: yes\n")
		}
		fmt.Println()

		model = glove.NewGloVe()

		if !*quiet {
			if *tokenize {
				fmt.Println("Reading and tokenizing corpus...")
			} else {
				fmt.Println("Reading pre-tokenized corpus...")
			}
		}
		tokens, err := readCorpusTokens(*corpusFile, *tokenize)
		if err != nil {
			log.Fatalf("Failed to read corpus: %v", err)
		}

		if !*quiet {
			fmt.Printf("Total tokens: %d\n", len(tokens))
			fmt.Println("Building vocabulary...")
		}
		if err := model.BuildVocab(tokens); err != nil {
			log.Fatal(err)
		}

		if !*quiet {
			fmt.Println("Building co-occurrence matrix...")
		}
		if err := model.BuildCooccurrenceMatrix(tokens, *windowSize); err != nil {
			log.Fatal(err)
		}
		if !*quiet {
			fmt.Printf("Vocabulary size: %d\n", model.VocabSize)
			fmt.Printf("Number of non-zero entries: %d\n", len(model.Cooccur))
		}

		model.InitializeParameters(*vectorSize)
	}

	if !*quiet {
		fmt.Println("Training GloVe model...")
	}

	// Create callback function that handles both progress display and periodic saving
	var callback glove.ProgressCallback
	if *quiet && (*saveState == "" || *saveInterval == 0) {
		// No callback needed - use original Train method
		model.Train(*iterations, *threads)
	} else {
		// Create callback function
		callback = createProgressCallback(model, *quiet, *saveState, *saveInterval)
		model.TrainWithCallback(*iterations, *threads, callback)
	}

	if !*quiet {
		fmt.Printf("Saving vectors to %s (mode: %s, format: %s)...\n", *outputFile, *saveMode, *outputFormat)
	}
	if err := model.SaveVectorsMode(*outputFile, saveModeEnum, outputFormatEnum, *saveHeader); err != nil {
		log.Fatal(err)
	}

	if *saveState != "" {
		if !*quiet {
			fmt.Printf("Saving model state to %s...\n", *saveState)
		}
		if err := model.SaveModelState(*saveState, true, true); err != nil {
			log.Fatalf("Failed to save model state: %v", err)
		}
	}

	fmt.Println("Training completed successfully!")

	if *runExamples {
		runExampleUsage(model, *similarityWord, *analogy, *topN, *topSimilar)
	}
}

// readCorpusTokens reads a corpus file and returns tokens
// If tokenize is true, applies GloVe tokenization to raw text
// If tokenize is false, expects whitespace-separated tokens (Stanford GloVe format)
func readCorpusTokens(filename string, tokenize bool) ([]string, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	if tokenize {
		// Apply GloVe tokenization to raw text
		wordFreqs := glove.Tokenize(f, 1)
		tokens := make([]string, 0)
		for _, wf := range wordFreqs {
			// Expand each word according to its frequency
			for i := 0; i < wf.Freq; i++ {
				tokens = append(tokens, wf.Word)
			}
		}
		return tokens, nil
	} else {
		// Read pre-tokenized corpus (whitespace-separated tokens)
		var tokens []string
		scanner := bufio.NewScanner(f)
		buf := make([]byte, 0, 1024*1024)
		scanner.Buffer(buf, 1024*1024)

		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}
			lineTokens := strings.Fields(line)
			tokens = append(tokens, lineTokens...)
		}

		if err := scanner.Err(); err != nil {
			return nil, err
		}

		return tokens, nil
	}
}

// createProgressCallback creates a callback function that handles progress display and periodic state saving
func createProgressCallback(model *glove.GloVe, quiet bool, saveStatePath string, saveInterval int) glove.ProgressCallback {
	return func(progress glove.TrainingProgress) {
		// Display progress if not in quiet mode (every 10 iterations, like before)
		if !quiet && progress.Iteration%10 == 0 {
			fmt.Printf("Iteration %d/%d, Cost: %.6f, Time: %v\n",
				progress.Iteration, progress.MaxIterations, progress.Cost, progress.TimeElapsed.Truncate(time.Second))
		}

		// Save state periodically if specified
		if saveStatePath != "" && saveInterval > 0 && progress.Iteration%saveInterval == 0 {
			// Create filename with iteration number
			baseExt := strings.Split(saveStatePath, ".")
			var filename string
			if len(baseExt) > 1 {
				filename = fmt.Sprintf("%s_iter_%d.%s", strings.Join(baseExt[:len(baseExt)-1], "."), progress.Iteration, baseExt[len(baseExt)-1])
			} else {
				filename = fmt.Sprintf("%s_iter_%d", saveStatePath, progress.Iteration)
			}

			if !quiet {
				fmt.Printf("Saving periodic state to %s...\n", filename)
			}

			// Save the model state
			if err := model.SaveModelState(filename, true, true); err != nil {
				fmt.Printf("Warning: Failed to save periodic state to %s: %v\n", filename, err)
			}
		}
	}
}

func runExampleUsage(model *glove.GloVe, similarityWord, analogyStr string, topN, topSimilar int) {
	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Println("EXAMPLE USAGE")
	fmt.Println(strings.Repeat("=", 50))

	parts := strings.Split(analogyStr, "::")
	if len(parts) != 2 {
		fmt.Printf("Invalid analogy format '%s'. Expected format: 'a:b::c'\n", analogyStr)
		return
	}

	leftParts := strings.Split(parts[0], ":")
	if len(leftParts) != 2 {
		fmt.Printf("Invalid analogy format '%s'. Expected format: 'a:b::c'\n", analogyStr)
		return
	}

	analogyA, analogyB, analogyC := leftParts[0], leftParts[1], parts[1]

	analogies := model.WordAnalogy(analogyA, analogyB, analogyC, topN)
	fmt.Printf("Analogy results for '%s:%s :: %s:?':\n", analogyA, analogyB, analogyC)
	for i, word := range analogies {
		fmt.Printf("%d. %s\n", i+1, word)
	}

	if vec, ok := model.GetWordVector(similarityWord); ok {
		type WordSim struct {
			Word string
			Sim  float64
		}

		var similarities []WordSim
		for word := range model.Vocab {
			if word == similarityWord {
				continue
			}
			wordVec, _ := model.GetWordVector(word)
			sim := glove.CosineSimilarity(vec, wordVec)
			similarities = append(similarities, WordSim{word, sim})
		}

		sort.Slice(similarities, func(i, j int) bool {
			return similarities[i].Sim > similarities[j].Sim
		})

		fmt.Printf("\nMost similar words to '%s':\n", similarityWord)
		for i := 0; i < topSimilar && i < len(similarities); i++ {
			fmt.Printf("%d. %s (%.4f)\n", i+1, similarities[i].Word, similarities[i].Sim)
		}
	} else {
		fmt.Printf("\nWord '%s' not found in vocabulary\n", similarityWord)
	}
}
