package main

import (
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
		corpusFile     = flag.String("corpus", "corpus.txt", "Path to the text corpus file")
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
		help           = flag.Bool("help", false, "Show usage information")
	)

	flag.Parse()

	if *help {
		fmt.Printf("GloVe Training Utility\n\n")
		fmt.Printf("Usage: %s [options]\n\n", os.Args[0])
		fmt.Printf("This utility trains word embeddings using the GloVe algorithm.\n\n")
		fmt.Printf("Options:\n")
		flag.PrintDefaults()
		fmt.Printf("\nExamples:\n")
		fmt.Printf("  %s -corpus text8.txt -vector-size 200 -iterations 50 -examples\n", os.Args[0])
		fmt.Printf("  %s -corpus data.txt -save-mode word-only -output-format binary\n", os.Args[0])
		fmt.Printf("  %s -corpus big.txt -save-header -output-format both\n", os.Args[0])
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
		if *saveHeader {
			fmt.Printf("  Include header: yes\n")
		}
		fmt.Println()

		model = glove.NewGloVe()

		if !*quiet {
			fmt.Println("Building vocabulary...")
		}
		if err := model.BuildVocab(*corpusFile); err != nil {
			log.Fatal(err)
		}

		if !*quiet {
			fmt.Println("Building co-occurrence matrix...")
		}
		if err := model.BuildCooccurrenceMatrix(*corpusFile, *windowSize); err != nil {
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
