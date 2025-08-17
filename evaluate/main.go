package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"

	"github.com/n0madic/go-glove"
)

func main() {
	var (
		vectorsFile    = flag.String("vectors", "glove_vectors.txt", "Path to the trained vectors file")
		similarityWord = flag.String("similarity-word", "computer", "Word to find similarities for")
		analogy        = flag.String("analogy", "king:queen::man", "Analogy in format 'a:b::c' to solve for 'a:b::c:?'")
		topN           = flag.Int("top-n", 5, "Number of analogy results to show")
		topSimilar     = flag.Int("top-similar", 10, "Number of similar words to show")
		help           = flag.Bool("help", false, "Show usage information")
	)

	flag.Parse()

	if *help {
		fmt.Printf("GloVe Model Evaluation Utility\n\n")
		fmt.Printf("Usage: %s [options]\n\n", os.Args[0])
		fmt.Printf("This utility evaluates a trained GloVe model by testing word analogies\n")
		fmt.Printf("and finding the most similar words to a given word.\n\n")
		fmt.Printf("Options:\n")
		flag.PrintDefaults()
		fmt.Printf("\nExamples:\n")
		fmt.Printf("  %s -vectors glove_vectors.txt -similarity-word computer -top-similar 5\n", os.Args[0])
		fmt.Printf("  %s -analogy king:queen::man -top-n 3\n", os.Args[0])
		fmt.Printf("  %s -vectors my_vectors.txt -analogy paris:france::london -top-n 1\n", os.Args[0])
		return
	}

	if _, err := os.Stat(*vectorsFile); os.IsNotExist(err) {
		log.Fatalf("Vectors file does not exist: %s", *vectorsFile)
	}

	fmt.Printf("Loading GloVe vectors from: %s\n", *vectorsFile)

	model := &glove.GloVe{}
	if err := model.LoadVectors(*vectorsFile); err != nil {
		log.Fatalf("Failed to load vectors: %v", err)
	}

	fmt.Printf("Loaded %d word vectors\n", model.VocabSize)
	fmt.Println()

	evaluateModel(model, *similarityWord, *analogy, *topN, *topSimilar)
}

func evaluateModel(model *glove.GloVe, similarityWord, analogyStr string, topN, topSimilar int) {
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("GLOVE MODEL EVALUATION")
	fmt.Println(strings.Repeat("=", 60))

	evaluateAnalogy(model, analogyStr, topN)
	fmt.Println()
	evaluateSimilarity(model, similarityWord, topSimilar)
}

func evaluateAnalogy(model *glove.GloVe, analogyStr string, topN int) {
	fmt.Printf("📊 ANALOGY TESTING\n")
	fmt.Println(strings.Repeat("-", 30))

	parts := strings.Split(analogyStr, "::")
	if len(parts) != 2 {
		fmt.Printf("❌ Invalid analogy format '%s'. Expected format: 'a:b::c'\n", analogyStr)
		return
	}

	leftParts := strings.Split(parts[0], ":")
	if len(leftParts) != 2 {
		fmt.Printf("❌ Invalid analogy format '%s'. Expected format: 'a:b::c'\n", analogyStr)
		return
	}

	analogyA, analogyB, analogyC := leftParts[0], leftParts[1], parts[1]

	if _, exists := model.Vocab[analogyA]; !exists {
		fmt.Printf("❌ Word '%s' not found in vocabulary\n", analogyA)
		return
	}
	if _, exists := model.Vocab[analogyB]; !exists {
		fmt.Printf("❌ Word '%s' not found in vocabulary\n", analogyB)
		return
	}
	if _, exists := model.Vocab[analogyC]; !exists {
		fmt.Printf("❌ Word '%s' not found in vocabulary\n", analogyC)
		return
	}

	analogies := model.WordAnalogy(analogyA, analogyB, analogyC, topN)
	fmt.Printf("Query: %s is to %s as %s is to ?\n", analogyA, analogyB, analogyC)
	fmt.Printf("Results:\n")

	if len(analogies) == 0 {
		fmt.Printf("  ❌ No analogy results found\n")
		return
	}

	for i, word := range analogies {
		fmt.Printf("  %d. %s\n", i+1, word)
	}
}

func evaluateSimilarity(model *glove.GloVe, similarityWord string, topSimilar int) {
	fmt.Printf("🔍 SIMILARITY TESTING\n")
	fmt.Println(strings.Repeat("-", 30))

	vec, ok := model.GetWordVector(similarityWord)
	if !ok {
		fmt.Printf("❌ Word '%s' not found in vocabulary\n", similarityWord)
		return
	}

	type WordSim struct {
		Word string
		Sim  float64
	}

	var similarities []WordSim
	for word := range model.Vocab {
		if word == similarityWord {
			continue
		}
		wordVec, exists := model.GetWordVector(word)
		if !exists {
			continue
		}
		sim := glove.CosineSimilarity(vec, wordVec)
		similarities = append(similarities, WordSim{word, sim})
	}

	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].Sim > similarities[j].Sim
	})

	fmt.Printf("Most similar words to '%s':\n", similarityWord)
	if len(similarities) == 0 {
		fmt.Printf("  ❌ No similar words found\n")
		return
	}

	for i := 0; i < topSimilar && i < len(similarities); i++ {
		fmt.Printf("  %d. %-15s (similarity: %.4f)\n",
			i+1, similarities[i].Word, similarities[i].Sim)
	}

	if len(similarities) > topSimilar {
		fmt.Printf("  ... and %d more words\n", len(similarities)-topSimilar)
	}
}
