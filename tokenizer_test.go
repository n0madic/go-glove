package glove

import (
	"strings"
	"testing"
)

func TestTokenize(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		minFreq     int
		expectedLen int
		checkWords  map[string]int // word -> expected frequency
	}{
		{
			name:        "Basic tokenization (GloVe-optimized: lowercase)",
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
		{
			name:        "GloVe feature: digits to zero",
			input:       "The year 1984 and number 2023 become 0000 and 0000",
			minFreq:     1,
			expectedLen: 6,
			checkWords: map[string]int{
				"the":    1,
				"year":   1,
				"0000":   4, // 1984, 2023, 0000 (2x) all become 0000
				"and":    2,
				"number": 1,
				"become": 1,
			},
		},
		{
			name:        "GloVe feature: URL and email replacement",
			input:       "Visit https://example.com or email user@domain.com for help",
			minFreq:     1,
			expectedLen: 7,
			checkWords: map[string]int{
				"visit":   1,
				"<url>":   1,
				"or":      1,
				"email":   1,
				"<email>": 1,
				"for":     1,
				"help":    1,
			},
		},
		{
			name:        "GloVe feature: case normalization",
			input:       "HELLO Hello hello",
			minFreq:     1,
			expectedLen: 1,
			checkWords: map[string]int{
				"hello": 3, // All variants become lowercase
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			result := Tokenize(reader, tt.minFreq)

			if len(result) != tt.expectedLen {
				t.Errorf("Tokenize() returned %d words, expected %d", len(result), tt.expectedLen)
			}

			// Check expected words and frequencies
			resultMap := make(map[string]int)
			for _, wf := range result {
				resultMap[wf.Word] = wf.Freq
			}

			for word, expectedFreq := range tt.checkWords {
				if actualFreq, exists := resultMap[word]; !exists || actualFreq != expectedFreq {
					t.Errorf("Tokenize() word '%s' frequency = %d, expected %d", word, actualFreq, expectedFreq)
				}
			}
		})
	}
}

func TestTokenizePunctuation(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		shouldFind []string // Tokens that should definitely be present
	}{
		{
			name:       "Sentence ending punctuation (lowercase)",
			input:      "Hello world.",
			shouldFind: []string{"hello", "world", "."},
		},
		{
			name:       "Commas and semicolons (lowercase)",
			input:      "apple, banana; cherry",
			shouldFind: []string{"apple", ",", "banana", ";", "cherry"},
		},
		{
			name:       "Basic punctuation separation (lowercase)",
			input:      "word!",
			shouldFind: []string{"word", "!"},
		},
		{
			name:       "GloVe feature: hyphen preservation",
			input:      "twenty-one self-contained",
			shouldFind: []string{"twenty-one", "self-contained"},
		},
		{
			name:       "Ellipsis normalization",
			input:      "Wait... let me think",
			shouldFind: []string{"wait", "...", "let", "me", "think"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			result := Tokenize(reader, 1)

			resultMap := make(map[string]int)
			for _, wf := range result {
				resultMap[wf.Word] = wf.Freq
			}

			for _, token := range tt.shouldFind {
				if _, exists := resultMap[token]; !exists {
					t.Errorf("Tokenize() missing expected token '%s'", token)
				}
			}
		})
	}
}

func TestTokenizeContractions(t *testing.T) {
	// Test GloVe-optimized PTB-style contraction splitting
	tests := []struct {
		name       string
		input      string
		shouldFind []string // Tokens that should be present
	}{
		{
			name:       "I'm contraction (PTB-style)",
			input:      "I'm happy",
			shouldFind: []string{"i", "'m", "happy"},
		},
		{
			name:       "Can't -> ca n't (PTB-style)",
			input:      "can't do it",
			shouldFind: []string{"ca", "n't", "do", "it"},
		},
		{
			name:       "Won't -> wo n't (PTB-style)",
			input:      "won't work",
			shouldFind: []string{"wo", "n't", "work"},
		},
		{
			name:       "General n't contractions",
			input:      "don't shouldn't",
			shouldFind: []string{"do", "n't", "should", "n't"},
		},
		{
			name:       "Other contractions (PTB-style)",
			input:      "I've you'll he'd",
			shouldFind: []string{"i", "'ve", "you", "'ll", "he", "'d"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			result := Tokenize(reader, 1)

			resultMap := make(map[string]int)
			for _, wf := range result {
				resultMap[wf.Word] = wf.Freq
			}

			for _, token := range tt.shouldFind {
				if _, exists := resultMap[token]; !exists {
					t.Errorf("Tokenize() missing expected token '%s'", token)
				}
			}
		})
	}
}

func TestTokenizeSpecialCases(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		shouldFind []string // Some tokens that should be present
	}{
		{
			name:       "Email addresses replaced with <email>",
			input:      "Contact user@example.com for support",
			shouldFind: []string{"contact", "<email>", "for", "support"},
		},
		{
			name:       "Mathematical symbols separated (with digit normalization)",
			input:      "2 + 3 = 5",
			shouldFind: []string{"0", "+", "0", "=", "0"},
		},
		{
			name:       "Dollar amounts preserved (with digit normalization)",
			input:      "Price $100",
			shouldFind: []string{"price", "$000"},
		},
		{
			name:       "URL replacement",
			input:      "Check https://example.com/page for details",
			shouldFind: []string{"check", "<url>", "for", "details"},
		},
		{
			name:       "Numbers with commas and dots preserved",
			input:      "The price is $1,234.56 exactly",
			shouldFind: []string{"the", "price", "is", "$0,000.00", "exactly"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			result := Tokenize(reader, 1)

			resultMap := make(map[string]int)
			for _, wf := range result {
				resultMap[wf.Word] = wf.Freq
			}

			for _, token := range tt.shouldFind {
				if _, exists := resultMap[token]; !exists {
					t.Errorf("Tokenize() missing expected token '%s'", token)
				}
			}
		})
	}
}

func TestTokenizeUnicodeCharacters(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		shouldFind []string
	}{
		{
			name:       "Unicode ellipsis conversion (lowercase)",
			input:      "Wait\u2026 let me think",
			shouldFind: []string{"wait", "...", "let", "me", "think"},
		},
		{
			name:       "Unicode quotes normalization (lowercase)",
			input:      "Hello \u201Cworld\u201D and \u2018test\u2019",
			shouldFind: []string{"hello", "\"", "world", "\"", "and", "'", "test", "'"},
		},
		{
			name:       "Unicode dashes normalization",
			input:      "en\u2013dash and em\u2014dash",
			shouldFind: []string{"en-dash", "and", "em", "--", "dash"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			result := Tokenize(reader, 1)

			resultMap := make(map[string]int)
			for _, wf := range result {
				resultMap[wf.Word] = wf.Freq
			}

			for _, token := range tt.shouldFind {
				if _, exists := resultMap[token]; !exists {
					t.Errorf("Tokenize() missing expected token '%s'", token)
				}
			}
		})
	}
}

func TestTokenizeSorting(t *testing.T) {
	input := "apple banana apple cherry banana apple"
	reader := strings.NewReader(input)
	result := Tokenize(reader, 1)

	// Expected: apple(3), banana(2), cherry(1)
	// Should be sorted by descending frequency, then alphabetically
	expected := []WordFreq{
		{Word: "apple", Freq: 3},
		{Word: "banana", Freq: 2},
		{Word: "cherry", Freq: 1},
	}

	if len(result) != len(expected) {
		t.Errorf("Tokenize() returned %d words, expected %d", len(result), len(expected))
		return
	}

	for i, exp := range expected {
		if result[i].Word != exp.Word || result[i].Freq != exp.Freq {
			t.Errorf("Tokenize() result[%d] = {%s, %d}, expected {%s, %d}",
				i, result[i].Word, result[i].Freq, exp.Word, exp.Freq)
		}
	}
}

func TestTokenizeAlphabeticalSorting(t *testing.T) {
	// Test alphabetical sorting when frequencies are equal
	input := "zebra alpha beta"
	reader := strings.NewReader(input)
	result := Tokenize(reader, 1)

	// All have frequency 1, should be sorted alphabetically
	expected := []WordFreq{
		{Word: "alpha", Freq: 1},
		{Word: "beta", Freq: 1},
		{Word: "zebra", Freq: 1},
	}

	if len(result) != len(expected) {
		t.Errorf("Tokenize() returned %d words, expected %d", len(result), len(expected))
		return
	}

	for i, exp := range expected {
		if result[i].Word != exp.Word || result[i].Freq != exp.Freq {
			t.Errorf("Tokenize() result[%d] = {%s, %d}, expected {%s, %d}",
				i, result[i].Word, result[i].Freq, exp.Word, exp.Freq)
		}
	}
}

func TestTokenizeMultiline(t *testing.T) {
	input := `First line with words.
Second line with more words.
Third line with even more words.`

	reader := strings.NewReader(input)
	result := Tokenize(reader, 1)

	// Check that we got tokens from all lines (expecting lowercase)
	foundTokens := make(map[string]bool)
	for _, wf := range result {
		foundTokens[wf.Word] = true
	}

	expectedTokens := []string{"first", "second", "third", "line", "with", "words", "more", "even"}
	for _, token := range expectedTokens {
		if !foundTokens[token] {
			t.Errorf("Tokenize() missing expected token '%s'", token)
		}
	}

	// Check that "line", "with", "words", "more" have correct frequencies
	tokenFreqs := make(map[string]int)
	for _, wf := range result {
		tokenFreqs[wf.Word] = wf.Freq
	}

	if tokenFreqs["line"] != 3 {
		t.Errorf("Tokenize() 'line' frequency = %d, expected 3", tokenFreqs["line"])
	}
	if tokenFreqs["with"] != 3 {
		t.Errorf("Tokenize() 'with' frequency = %d, expected 3", tokenFreqs["with"])
	}
	if tokenFreqs["words"] != 3 {
		t.Errorf("Tokenize() 'words' frequency = %d, expected 3", tokenFreqs["words"])
	}
	if tokenFreqs["more"] != 2 {
		t.Errorf("Tokenize() 'more' frequency = %d, expected 2", tokenFreqs["more"])
	}
}

func TestTokenizeHelperFunction(t *testing.T) {
	// Test the convenience wrapper function
	input := "test test test test test"
	reader := strings.NewReader(input)
	result := Tokenize(reader, 0)

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
func BenchmarkTokenize(b *testing.B) {
	input := "The quick brown fox jumps over the lazy dog. " +
		"This is a sample text for benchmarking tokenization performance. " +
		"It contains various punctuation marks, numbers like 123, and contractions like don't."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reader := strings.NewReader(input)
		Tokenize(reader, 1)
	}
}

func TestTokenizeWithOptions(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		options     TokenizerOptions
		minFreq     int
		expectedLen int
		checkWords  map[string]int
	}{
		{
			name:        "Disable lowercase",
			input:       "Hello WORLD hello",
			options:     TokenizerOptions{Lowercase: false, DigitsToZero: false, ReplaceURLs: false, ReplaceEmails: false, KeepHyphens: true},
			minFreq:     1,
			expectedLen: 3,
			checkWords: map[string]int{
				"Hello": 1,
				"WORLD": 1,
				"hello": 1,
			},
		},
		{
			name:        "Disable digits to zero",
			input:       "The year 2023 and 2024",
			options:     TokenizerOptions{Lowercase: true, DigitsToZero: false, ReplaceURLs: false, ReplaceEmails: false, KeepHyphens: true},
			minFreq:     1,
			expectedLen: 5,
			checkWords: map[string]int{
				"the":  1,
				"year": 1,
				"2023": 1,
				"and":  1,
				"2024": 1,
			},
		},
		{
			name:        "Disable URL replacement",
			input:       "Visit https://example.com today",
			options:     TokenizerOptions{Lowercase: true, DigitsToZero: true, ReplaceURLs: false, ReplaceEmails: false, KeepHyphens: true},
			minFreq:     1,
			expectedLen: 5,
			checkWords: map[string]int{
				"visit":         1,
				"https":         1,
				":":             1,
				"//example.com": 1,
				"today":         1,
			},
		},
		{
			name:        "Split hyphens",
			input:       "twenty-one self-contained",
			options:     TokenizerOptions{Lowercase: true, DigitsToZero: true, ReplaceURLs: true, ReplaceEmails: true, KeepHyphens: false},
			minFreq:     1,
			expectedLen: 5,
			checkWords: map[string]int{
				"twenty":    1,
				"-":         2, // Appears twice
				"one":       1,
				"self":      1,
				"contained": 1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			result := TokenizeWithOptions(reader, tt.minFreq, tt.options)

			if len(result) != tt.expectedLen {
				t.Errorf("TokenizeWithOptions() returned %d words, expected %d", len(result), tt.expectedLen)
			}

			// Check expected words and frequencies
			resultMap := make(map[string]int)
			for _, wf := range result {
				resultMap[wf.Word] = wf.Freq
			}

			for word, expectedFreq := range tt.checkWords {
				if actualFreq, exists := resultMap[word]; !exists || actualFreq != expectedFreq {
					t.Errorf("TokenizeWithOptions() word '%s' frequency = %d, expected %d", word, actualFreq, expectedFreq)
				}
			}
		})
	}
}

func TestSmartApostrophesContractions(t *testing.T) {
	// Test critical fix: smart apostrophes should preserve contractions
	input := "I\u2019m sure you\u2019ll don\u2019t can\u2019t won\u2019t"
	reader := strings.NewReader(input)
	result := Tokenize(reader, 1)

	resultMap := make(map[string]int)
	for _, wf := range result {
		resultMap[wf.Word] = wf.Freq
	}

	expectedTokens := []string{"i", "'m", "sure", "you", "'ll", "do", "n't", "ca", "n't", "wo", "n't"}
	for _, token := range expectedTokens {
		if _, exists := resultMap[token]; !exists {
			t.Errorf("SmartApostrophes test missing expected token '%s'", token)
		}
	}
}

func TestAdvancedNumberHandling(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		shouldFind []string
	}{
		{
			name:       "Multi-comma numbers",
			input:      "The amount is 1,234,567.89 dollars",
			shouldFind: []string{"the", "amount", "is", "0,000,000.00", "dollars"},
		},
		{
			name:       "IP addresses",
			input:      "Connect to 127.0.0.1 server",
			shouldFind: []string{"connect", "to", "000.0.0.0", "server"},
		},
		{
			name:       "Unary minus with Unicode minus sign",
			input:      "Temperature is \u221242 degrees",
			shouldFind: []string{"temperature", "is", "-00", "degrees"},
		},
		{
			name:       "Mixed currency and decimals",
			input:      "Price: $1,299.99 (was €1,500.00)",
			shouldFind: []string{"price", ":", "$0,000.00", "(", "was", "€0,000.00", ")"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			result := Tokenize(reader, 1)

			resultMap := make(map[string]int)
			for _, wf := range result {
				resultMap[wf.Word] = wf.Freq
			}

			for _, token := range tt.shouldFind {
				if _, exists := resultMap[token]; !exists {
					t.Errorf("Advanced number test '%s' missing expected token '%s'", tt.name, token)
				}
			}
		})
	}
}

func TestGloVeOptimizedFeatures(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected map[string]int
	}{
		{
			name:  "Mixed case normalization with smart quotes",
			input: "Hello \u201CWorld\u201D and \u2018Test\u2019 said John",
			expected: map[string]int{
				"hello": 1,
				"\"":    2, // Double quotes
				"world": 1,
				"and":   1,
				"'":     2, // Single quotes/apostrophes
				"test":  1,
				"said":  1,
				"john":  1,
			},
		},
		{
			name:  "Complex contractions with digits",
			input: "I\u2019m from the \u201990s, you\u2019re from 2000s",
			expected: map[string]int{
				"i":     1,
				"'m":    1,
				"from":  2,
				"the":   1,
				"'00s":  1, // digits normalized: '90s -> '00s
				",":     1,
				"you":   1,
				"'re":   1,
				"0000s": 1, // 2000s -> 0000s
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			result := Tokenize(reader, 1)

			resultMap := make(map[string]int)
			for _, wf := range result {
				resultMap[wf.Word] = wf.Freq
			}

			for word, expectedFreq := range tt.expected {
				if actualFreq, exists := resultMap[word]; !exists || actualFreq != expectedFreq {
					t.Errorf("GloVe optimized test '%s' word '%s' frequency = %d, expected %d", tt.name, word, actualFreq, expectedFreq)
				}
			}
		})
	}
}

func BenchmarkTokenizeLargeText(b *testing.B) {
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
		Tokenize(reader, 5)
	}
}
