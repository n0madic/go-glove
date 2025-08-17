package glove

import (
	"bufio"
	"io"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

type WordFreq struct {
	Word string
	Freq int
}

// StanfordTokenize implements tokenization in the style of Stanford PTBTokenizer
func StanfordTokenize(reader io.Reader, minFreq int) []WordFreq {
	if minFreq <= 0 {
		minFreq = 5
	}

	scanner := bufio.NewScanner(reader)
	scanner.Split(bufio.ScanLines)

	wordCount := make(map[string]int)

	for scanner.Scan() {
		line := scanner.Text()
		tokens := tokenizeLine(line)
		for _, token := range tokens {
			if token != "" {
				wordCount[token]++
			}
		}
	}

	// Filter by minimum frequency and create result
	var result []WordFreq
	for word, freq := range wordCount {
		if freq >= minFreq {
			result = append(result, WordFreq{Word: word, Freq: freq})
		}
	}

	// Sort by descending frequency, for equal frequency - alphabetically
	sort.Slice(result, func(i, j int) bool {
		if result[i].Freq != result[j].Freq {
			return result[i].Freq > result[j].Freq
		}
		return result[i].Word < result[j].Word
	})

	return result
}

// tokenizeLine tokenizes a single line according to Stanford Tokenizer rules
func tokenizeLine(text string) []string {
	if text == "" {
		return []string{}
	}

	// Preprocess text
	text = preprocessText(text)

	// Main patterns for tokenization
	patterns := []struct {
		pattern *regexp.Regexp
		replace string
	}{
		// Separate sentence-ending punctuation
		{regexp.MustCompile(`([.!?])(\s+|$)`), " $1 "},

		// Separate commas, semicolons, colons
		{regexp.MustCompile(`([,:;])`), " $1 "},

		// Separate opening brackets and quotes
		{regexp.MustCompile(`([({\["'«])`), " $1 "},

		// Separate closing brackets and quotes
		{regexp.MustCompile(`([)}\]"'»])`), " $1 "},

		// Handle contractions with apostrophe
		{regexp.MustCompile(`(?i)\b(can)'t\b`), "$1 not"},
		{regexp.MustCompile(`(?i)\b(won)'t\b`), "will not"},
		{regexp.MustCompile(`(?i)\b(n)'t\b`), " not"},
		{regexp.MustCompile(`(?i)\b('m)\b`), " am"},
		{regexp.MustCompile(`(?i)\b('re)\b`), " are"},
		{regexp.MustCompile(`(?i)\b('ve)\b`), " have"},
		{regexp.MustCompile(`(?i)\b('ll)\b`), " will"},
		{regexp.MustCompile(`(?i)\b('d)\b`), " would"},
		{regexp.MustCompile(`(?i)\b('s)\b`), " 's"},

		// Separate hyphens between words (except for compound words)
		{regexp.MustCompile(`(\w)-(\w)`), "$1 - $2"},

		// Separate ellipsis
		{regexp.MustCompile(`\.{2,}`), " ... "},

		// Separate dashes
		{regexp.MustCompile(`--+`), " -- "},

		// Percent signs and currency
		{regexp.MustCompile(`([%$€£¥₽])`), " $1 "},

		// Mathematical symbols
		{regexp.MustCompile(`([+\-*/=<>])`), " $1 "},

		// Ampersand
		{regexp.MustCompile(`(&)`), " $1 "},
	}

	for _, p := range patterns {
		text = p.pattern.ReplaceAllString(text, p.replace)
	}

	// Special handling for numbers with commas and dots
	text = processNumbers(text)

	// Split into tokens by spaces
	tokens := strings.Fields(text)

	// Postprocess tokens
	var result []string
	for _, token := range tokens {
		token = strings.TrimSpace(token)
		if token != "" {
			// Additional processing for special cases
			processedTokens := postprocessToken(token)
			result = append(result, processedTokens...)
		}
	}

	return result
}

// preprocessText performs preliminary text processing
func preprocessText(text string) string {
	// Normalize spaces
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")

	// Replace special Unicode characters with ASCII equivalents
	replacements := map[string]string{
		"“": "\"",
		"”": "\"",
		"‘": "'",
		"’": "'",
		"–": "-",
		"—": "--",
		"…": "...",
	}

	for old, new := range replacements {
		text = strings.ReplaceAll(text, old, new)
	}

	return text
}

// processNumbers handles numbers with separators
func processNumbers(text string) string {
	// Pattern for numbers with commas (1,000,000)
	commaNumber := regexp.MustCompile(`\b(\d{1,3}(?:,\d{3})+)\b`)
	text = commaNumber.ReplaceAllStringFunc(text, func(s string) string {
		return strings.ReplaceAll(s, ",", "")
	})

	// Pattern for decimal numbers (3.14)
	decimalNumber := regexp.MustCompile(`\b(\d+)\.(\d+)\b`)
	text = decimalNumber.ReplaceAllString(text, "$1.$2")

	return text
}

// postprocessToken performs final processing of a single token
func postprocessToken(token string) []string {
	// Check if token is a URL or email
	if isURL(token) || isEmail(token) {
		return []string{token}
	}

	// Check for acronyms (U.S.A., Ph.D.)
	if isAcronym(token) {
		return []string{token}
	}

	// Handle possessive case
	if strings.HasSuffix(token, "'s") || strings.HasSuffix(token, "'S") {
		base := token[:len(token)-2]
		if base != "" {
			return []string{base, "'s"}
		}
	}

	// If token contains only punctuation, return as is
	if isAllPunctuation(token) {
		return []string{token}
	}

	// Split glued words with punctuation at the ends
	if len(token) > 1 {
		firstRune := rune(token[0])
		lastRune := rune(token[len(token)-1])

		if unicode.IsPunct(firstRune) && !unicode.IsPunct(rune(token[1])) {
			return []string{string(firstRune), token[1:]}
		}

		if unicode.IsPunct(lastRune) && !unicode.IsPunct(rune(token[len(token)-2])) {
			return []string{token[:len(token)-1], string(lastRune)}
		}
	}

	return []string{token}
}

// isURL checks if the token is a URL
func isURL(token string) bool {
	urlPattern := regexp.MustCompile(`^(https?|ftp)://[^\s/$.?#].[^\s]*$`)
	return urlPattern.MatchString(token)
}

// isEmail checks if the token is an email
func isEmail(token string) bool {
	emailPattern := regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)
	return emailPattern.MatchString(token)
}

// isAcronym checks if the token is an acronym
func isAcronym(token string) bool {
	acronymPattern := regexp.MustCompile(`^([A-Z]\.)+[A-Z]?\.?$`)
	return acronymPattern.MatchString(token)
}

// isAllPunctuation checks if the token consists only of punctuation
func isAllPunctuation(token string) bool {
	for _, r := range token {
		if !unicode.IsPunct(r) && !unicode.IsSymbol(r) {
			return false
		}
	}
	return true
}

// Helper function for convenient use with default parameters
func Tokenize(reader io.Reader) []WordFreq {
	return StanfordTokenize(reader, 5)
}
