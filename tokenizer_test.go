package glove

import (
	"strings"
	"testing"
)

func TestStanfordTokenize(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		minFreq     int
		expectedLen int
		checkWords  map[string]int // word -> expected frequency
	}{
		{
			name:        "Basic tokenization",
			input:       "hello world hello",
			minFreq:     1,
			expectedLen: 2,
			checkWords: map[string]int{
				"hello": 2,
				"world": 1,
			},
		},
		{
			name:        "Frequency filtering",
			input:       "hello hello hello world world test",
			minFreq:     2,
			expectedLen: 2,
			checkWords: map[string]int{
				"hello": 3,
				"world": 2,
			},
		},
		{
			name:        "Empty input",
			input:       "",
			minFreq:     1,
			expectedLen: 0,
			checkWords:  map[string]int{},
		},
		{
			name:        "Default minFreq with zero",
			input:       "test test test test test",
			minFreq:     0, // Should default to 5
			expectedLen: 1,
			checkWords: map[string]int{
				"test": 5,
			},
		},
		{
			name:        "Default minFreq with negative",
			input:       "word word word word word",
			minFreq:     -1, // Should default to 5
			expectedLen: 1,
			checkWords: map[string]int{
				"word": 5,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			result := StanfordTokenize(reader, tt.minFreq)

			if len(result) != tt.expectedLen {
				t.Errorf("StanfordTokenize() returned %d words, expected %d", len(result), tt.expectedLen)
			}

			// Check expected words and frequencies
			resultMap := make(map[string]int)
			for _, wf := range result {
				resultMap[wf.Word] = wf.Freq
			}

			for word, expectedFreq := range tt.checkWords {
				if actualFreq, exists := resultMap[word]; !exists || actualFreq != expectedFreq {
					t.Errorf("StanfordTokenize() word '%s' frequency = %d, expected %d", word, actualFreq, expectedFreq)
				}
			}
		})
	}
}

func TestStanfordTokenizePunctuation(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		shouldFind []string // Tokens that should definitely be present
	}{
		{
			name:       "Sentence ending punctuation",
			input:      "Hello world.",
			shouldFind: []string{"Hello", "world", "."},
		},
		{
			name:       "Commas and semicolons",
			input:      "apple, banana; cherry",
			shouldFind: []string{"apple", ",", "banana", ";", "cherry"},
		},
		{
			name:       "Basic punctuation separation",
			input:      "word!",
			shouldFind: []string{"word", "!"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			result := StanfordTokenize(reader, 1)

			resultMap := make(map[string]int)
			for _, wf := range result {
				resultMap[wf.Word] = wf.Freq
			}

			for _, token := range tt.shouldFind {
				if _, exists := resultMap[token]; !exists {
					t.Errorf("StanfordTokenize() missing expected token '%s'", token)
				}
			}
		})
	}
}

func TestStanfordTokenizeContractions(t *testing.T) {
	// Test how the tokenizer actually handles contractions (it splits them but doesn't expand)
	tests := []struct {
		name       string
		input      string
		shouldFind []string // Tokens that should be present
	}{
		{
			name:       "Apostrophe splitting",
			input:      "I'm happy",
			shouldFind: []string{"I", "'", "m", "happy"},
		},
		{
			name:       "Can't splitting",
			input:      "can't do",
			shouldFind: []string{"can", "'", "t", "do"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			result := StanfordTokenize(reader, 1)

			resultMap := make(map[string]int)
			for _, wf := range result {
				resultMap[wf.Word] = wf.Freq
			}

			for _, token := range tt.shouldFind {
				if _, exists := resultMap[token]; !exists {
					t.Errorf("StanfordTokenize() missing expected token '%s'", token)
				}
			}
		})
	}
}

func TestStanfordTokenizeSpecialCases(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		shouldFind []string // Some tokens that should be present
	}{
		{
			name:       "Email addresses preserved",
			input:      "Contact user@example.com for support",
			shouldFind: []string{"Contact", "user@example.com", "for", "support"},
		},
		{
			name:       "Mathematical symbols separated",
			input:      "2 + 3 = 5",
			shouldFind: []string{"2", "+", "3", "=", "5"},
		},
		{
			name:       "Dollar signs separated",
			input:      "Price $100",
			shouldFind: []string{"Price", "$", "100"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			result := StanfordTokenize(reader, 1)

			resultMap := make(map[string]int)
			for _, wf := range result {
				resultMap[wf.Word] = wf.Freq
			}

			for _, token := range tt.shouldFind {
				if _, exists := resultMap[token]; !exists {
					t.Errorf("StanfordTokenize() missing expected token '%s'", token)
				}
			}
		})
	}
}

func TestStanfordTokenizeUnicodeCharacters(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		shouldFind []string
	}{
		{
			name:       "Unicode ellipsis conversion",
			input:      "Wait… let me think",
			shouldFind: []string{"Wait", "...", "let", "me", "think"},
		},
		{
			name:       "Basic Unicode handling",
			input:      "Hello \"world\"",
			shouldFind: []string{"Hello", "world"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			result := StanfordTokenize(reader, 1)

			resultMap := make(map[string]int)
			for _, wf := range result {
				resultMap[wf.Word] = wf.Freq
			}

			for _, token := range tt.shouldFind {
				if _, exists := resultMap[token]; !exists {
					t.Errorf("StanfordTokenize() missing expected token '%s'", token)
				}
			}
		})
	}
}

func TestStanfordTokenizeSorting(t *testing.T) {
	input := "apple banana apple cherry banana apple"
	reader := strings.NewReader(input)
	result := StanfordTokenize(reader, 1)

	// Expected: apple(3), banana(2), cherry(1)
	// Should be sorted by descending frequency, then alphabetically
	expected := []WordFreq{
		{Word: "apple", Freq: 3},
		{Word: "banana", Freq: 2},
		{Word: "cherry", Freq: 1},
	}

	if len(result) != len(expected) {
		t.Errorf("StanfordTokenize() returned %d words, expected %d", len(result), len(expected))
		return
	}

	for i, exp := range expected {
		if result[i].Word != exp.Word || result[i].Freq != exp.Freq {
			t.Errorf("StanfordTokenize() result[%d] = {%s, %d}, expected {%s, %d}",
				i, result[i].Word, result[i].Freq, exp.Word, exp.Freq)
		}
	}
}

func TestStanfordTokenizeAlphabeticalSorting(t *testing.T) {
	// Test alphabetical sorting when frequencies are equal
	input := "zebra alpha beta"
	reader := strings.NewReader(input)
	result := StanfordTokenize(reader, 1)

	// All have frequency 1, should be sorted alphabetically
	expected := []WordFreq{
		{Word: "alpha", Freq: 1},
		{Word: "beta", Freq: 1},
		{Word: "zebra", Freq: 1},
	}

	if len(result) != len(expected) {
		t.Errorf("StanfordTokenize() returned %d words, expected %d", len(result), len(expected))
		return
	}

	for i, exp := range expected {
		if result[i].Word != exp.Word || result[i].Freq != exp.Freq {
			t.Errorf("StanfordTokenize() result[%d] = {%s, %d}, expected {%s, %d}",
				i, result[i].Word, result[i].Freq, exp.Word, exp.Freq)
		}
	}
}

func TestStanfordTokenizeMultiline(t *testing.T) {
	input := `First line with words.
Second line with more words.
Third line with even more words.`

	reader := strings.NewReader(input)
	result := StanfordTokenize(reader, 1)

	// Check that we got tokens from all lines
	foundTokens := make(map[string]bool)
	for _, wf := range result {
		foundTokens[wf.Word] = true
	}

	expectedTokens := []string{"First", "Second", "Third", "line", "with", "words", "more", "even"}
	for _, token := range expectedTokens {
		if !foundTokens[token] {
			t.Errorf("StanfordTokenize() missing expected token '%s'", token)
		}
	}

	// Check that "line", "with", "words", "more" have correct frequencies
	tokenFreqs := make(map[string]int)
	for _, wf := range result {
		tokenFreqs[wf.Word] = wf.Freq
	}

	if tokenFreqs["line"] != 3 {
		t.Errorf("StanfordTokenize() 'line' frequency = %d, expected 3", tokenFreqs["line"])
	}
	if tokenFreqs["with"] != 3 {
		t.Errorf("StanfordTokenize() 'with' frequency = %d, expected 3", tokenFreqs["with"])
	}
	if tokenFreqs["words"] != 3 {
		t.Errorf("StanfordTokenize() 'words' frequency = %d, expected 3", tokenFreqs["words"])
	}
	if tokenFreqs["more"] != 2 {
		t.Errorf("StanfordTokenize() 'more' frequency = %d, expected 2", tokenFreqs["more"])
	}
}

func TestTokenizeHelperFunction(t *testing.T) {
	// Test the convenience wrapper function
	input := "test test test test test"
	reader := strings.NewReader(input)
	result := Tokenize(reader)

	// Should use default minFreq of 5
	expected := []WordFreq{
		{Word: "test", Freq: 5},
	}

	if len(result) != len(expected) {
		t.Errorf("Tokenize() returned %d words, expected %d", len(result), len(expected))
		return
	}

	if result[0].Word != expected[0].Word || result[0].Freq != expected[0].Freq {
		t.Errorf("Tokenize() result = {%s, %d}, expected {%s, %d}",
			result[0].Word, result[0].Freq, expected[0].Word, expected[0].Freq)
	}
}

// Benchmark tests for tokenization performance
func BenchmarkStanfordTokenize(b *testing.B) {
	input := "The quick brown fox jumps over the lazy dog. " +
		"This is a sample text for benchmarking tokenization performance. " +
		"It contains various punctuation marks, numbers like 123, and contractions like don't."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reader := strings.NewReader(input)
		StanfordTokenize(reader, 1)
	}
}

func BenchmarkStanfordTokenizeLargeText(b *testing.B) {
	// Create a larger text for benchmarking
	baseText := "The quick brown fox jumps over the lazy dog. "
	var largeText strings.Builder
	for i := 0; i < 1000; i++ {
		largeText.WriteString(baseText)
	}

	input := largeText.String()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reader := strings.NewReader(input)
		StanfordTokenize(reader, 5)
	}
}
