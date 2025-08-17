package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"github.com/n0madic/go-glove"
)

func main() {
	var (
		inputFile     = flag.String("input", "", "Input text file to tokenize (use - for stdin)")
		outputFile    = flag.String("output", "", "Output file for tokenized text (use - for stdout)")
		lowercase     = flag.Bool("lowercase", true, "Convert text to lowercase")
		digitsToZero  = flag.Bool("digits-to-zero", true, "Normalize all digits to 0")
		replaceURLs   = flag.Bool("replace-urls", true, "Replace URLs with <url> token")
		replaceEmails = flag.Bool("replace-emails", true, "Replace emails with <email> token")
		keepHyphens   = flag.Bool("keep-hyphens", true, "Keep hyphenated words intact")
		minFreq       = flag.Int("min-freq", 1, "Minimum word frequency to include")
		showFreqs     = flag.Bool("show-freqs", false, "Output word frequencies instead of tokens")
		help          = flag.Bool("help", false, "Show usage information")
	)

	flag.Parse()

	if *help {
		fmt.Printf("GloVe Tokenization Utility\n\n")
		fmt.Printf("Usage: %s [options]\n\n", os.Args[0])
		fmt.Printf("This utility tokenizes text using GloVe-optimized preprocessing.\n")
		fmt.Printf("It performs PTB-style tokenization with features like contraction splitting,\n")
		fmt.Printf("Unicode normalization, and optional digit/URL/email replacement.\n\n")
		fmt.Printf("Options:\n")
		flag.PrintDefaults()
		fmt.Printf("\nExamples:\n")
		fmt.Printf("  # Tokenize a file with default GloVe settings\n")
		fmt.Printf("  %s -input corpus.txt -output tokenized.txt\n\n", os.Args[0])
		fmt.Printf("  # Tokenize stdin and output to stdout\n")
		fmt.Printf("  cat text.txt | %s -input - -output -\n\n", os.Args[0])
		fmt.Printf("  # Generate word frequency list\n")
		fmt.Printf("  %s -input corpus.txt -show-freqs -min-freq 5 > vocab.txt\n\n", os.Args[0])
		fmt.Printf("  # Tokenize with custom options (preserve case, don't replace URLs)\n")
		fmt.Printf("  %s -input text.txt -lowercase=false -replace-urls=false\n\n", os.Args[0])
		return
	}

	if *inputFile == "" {
		log.Fatal("Input file must be specified. Use -input filename or -input - for stdin")
	}

	// Open input
	var input io.Reader
	if *inputFile == "-" {
		input = os.Stdin
	} else {
		f, err := os.Open(*inputFile)
		if err != nil {
			log.Fatalf("Failed to open input file: %v", err)
		}
		defer f.Close()
		input = f
	}

	// Configure tokenization options
	options := glove.TokenizerOptions{
		Lowercase:       *lowercase,
		DigitsToZero:    *digitsToZero,
		ReplaceURLs:     *replaceURLs,
		ReplaceEmails:   *replaceEmails,
		KeepHyphens:     *keepHyphens,
		SentencePerLine: false,
	}

	// Tokenize
	wordFreqs := glove.TokenizeWithOptions(input, *minFreq, options)

	// Open output
	var output io.Writer
	if *outputFile == "" || *outputFile == "-" {
		output = os.Stdout
	} else {
		f, err := os.Create(*outputFile)
		if err != nil {
			log.Fatalf("Failed to create output file: %v", err)
		}
		defer f.Close()
		output = f
	}

	// Write output
	if *showFreqs {
		// Output word frequency format: word frequency
		for _, wf := range wordFreqs {
			fmt.Fprintf(output, "%s %d\n", wf.Word, wf.Freq)
		}
	} else {
		// Output tokens separated by spaces (Stanford GloVe format)
		var tokens []string
		for _, wf := range wordFreqs {
			// Repeat each word according to its frequency
			for i := 0; i < wf.Freq; i++ {
				tokens = append(tokens, wf.Word)
			}
		}
		fmt.Fprint(output, strings.Join(tokens, " "))
		if len(tokens) > 0 {
			fmt.Fprint(output, "\n")
		}
	}
}
