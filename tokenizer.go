package glove

import (
	"bufio"
	"io"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

type WordFreq struct {
	Word string
	Freq int
}

// TokenizerOptions configures the tokenization behavior
type TokenizerOptions struct {
	Lowercase       bool
	DigitsToZero    bool
	ReplaceURLs     bool
	ReplaceEmails   bool
	KeepHyphens     bool
	SentencePerLine bool
}

// DefaultGloVeOptions returns tokenization options optimized for GloVe training
func DefaultGloVeOptions() TokenizerOptions {
	return TokenizerOptions{
		Lowercase:       true,  // For compact vocabulary
		DigitsToZero:    true,  // Normalize digits to 0
		ReplaceURLs:     true,  // Replace URLs with <url>
		ReplaceEmails:   true,  // Replace emails with <email>
		KeepHyphens:     true,  // Preserve hyphenated words
		SentencePerLine: false, // Stream tokens continuously
	}
}

var (
	// URLs / emails
	reURL   = regexp.MustCompile(`(?i)\b(?:https?|ftp)://[^\s]+`)
	reEmail = regexp.MustCompile(`(?i)\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b`)

	// Ellipsis (3+ dots) -> " ... "
	reEllipsis = regexp.MustCompile(`\.{3,}`)

	// Normalize spacings around selected ASCII punctuation (except dot/comma which need care)
	rePunctSpace = regexp.MustCompile(`([;:!?(){}\[\]"])`)

	// Contractions (special PTB-ish cases first)
	reWonT   = regexp.MustCompile(`\b(?i)won't\b`)
	reCanT   = regexp.MustCompile(`\b(?i)can't\b`)
	reShanT  = regexp.MustCompile(`\b(?i)shan't\b`)
	reNt     = regexp.MustCompile(`\b([A-Za-z]+)n't\b`)
	reAposRe = regexp.MustCompile(`\b([A-Za-z]+)'(re|ve|ll|d|m|s)\b`)

	// After spacing commas/dots, re-join numeric patterns: 1 , 234  -> 1,234 ;  3 . 14 -> 3.14
	reNumCommaFix = regexp.MustCompile(`(\d)\s*,\s*(\d)`)
	reNumDotFix   = regexp.MustCompile(`(\d)\s*\.\s*(\d)`)

	// Space terminal sentence punctuation . ! ?  (but we will later re-join decimals)
	reTermPunct = regexp.MustCompile(`([^.0-9A-Za-z])([.!?])|([A-Za-z0-9])([.!?])(\s|$)`)

	// Space double dash as a separate token
	reDoubleDash = regexp.MustCompile(`--`)
)

// Tokenize implements tokenization optimized for GloVe training
func Tokenize(reader io.Reader, minFreq int) []WordFreq {
	return TokenizeWithOptions(reader, minFreq, DefaultGloVeOptions())
}

// TokenizeWithOptions provides configurable tokenization
func TokenizeWithOptions(reader io.Reader, minFreq int, options TokenizerOptions) []WordFreq {
	if minFreq <= 0 {
		minFreq = 5
	}

	scanner := bufio.NewScanner(reader)
	scanner.Split(bufio.ScanLines)

	// Increase buffer to handle very long lines
	const maxCapacity = 1024 * 1024 // 1MB
	buf := make([]byte, 64*1024)
	scanner.Buffer(buf, maxCapacity)

	wordCount := make(map[string]int)

	for scanner.Scan() {
		line := scanner.Text()
		if !utf8.ValidString(line) {
			// Replace invalid UTF-8 runes
			line = strings.ToValidUTF8(line, " ")
		}

		tokenizedLine := tokenizeLine(line, options)
		if tokenizedLine == "" {
			continue
		}

		tokens := strings.Fields(tokenizedLine)
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

// tokenizeLine tokenizes a single line with GloVe-optimized processing
func tokenizeLine(line string, opt TokenizerOptions) string {
	// Normalize unicode punctuation
	line = normalizeUnicode(line)

	// Replace URL/email with special tokens
	line = replaceSpecials(line, opt)

	// PTB-like contractions
	line = splitContractions(line)

	// Space punctuation without breaking decimals/hyphens
	line = spacePunctuation(line, opt.KeepHyphens)

	// Normalize digits
	if opt.DigitsToZero {
		line = digitsToZero(line)
	}

	// Lowercase
	line = toLowerIfNeeded(line, opt.Lowercase)

	// Collapse spaces
	line = collapseSpaces(line)
	return line
}

// Basic unicode-like normalization for quotes/dashes/ellipsis (ASCII-safe)
func normalizeUnicode(s string) string {
	replacer := strings.NewReplacer(
		// Double quotes
		"\u201C", `"`, "\u201D", `"`, "\u201E", `"`, "\u00AB", `"`, "\u00BB", `"`,
		// Single quotes -> ASCII apostrophe to preserve contractions
		"\u2018", "'", "\u2019", "'", "\u201A", "'",
		// Dashes
		"\u2014", "--", "\u2013", "-", "\u2212", "-",
		// Ellipsis
		"\u2026", "...",
	)
	return replacer.Replace(s)
}

func replaceSpecials(s string, opt TokenizerOptions) string {
	if opt.ReplaceURLs {
		s = reURL.ReplaceAllString(s, " <url> ")
	}
	if opt.ReplaceEmails {
		s = reEmail.ReplaceAllString(s, " <email> ")
	}
	return s
}

// PTB-like contraction splitting without lemmatization
func splitContractions(s string) string {
	// Special cases to match PTB-ish tokens
	s = reWonT.ReplaceAllStringFunc(s, func(m string) string {
		// Preserve case of first letter, but produce tokens "wo n't"
		if len(m) >= 1 && (m[0] == 'W' || m[0] == 'w') {
			return "wo n't"
		}
		return "wo n't"
	})
	s = reCanT.ReplaceAllStringFunc(s, func(m string) string {
		// "can't" -> "ca n't"
		return "ca n't"
	})
	s = reShanT.ReplaceAllStringFunc(s, func(m string) string {
		// "shan't" -> "sha n't"
		return "sha n't"
	})
	// General n't
	s = reNt.ReplaceAllString(s, "${1} n't")
	// 're 've 'll 'd 'm 's
	s = reAposRe.ReplaceAllString(s, "${1} '$2")
	return s
}

// Space punctuation carefully; do not break decimals and hyphenated words
func spacePunctuation(s string, keepHyphens bool) string {
	// Space double dash as a token
	s = reDoubleDash.ReplaceAllString(s, " -- ")

	// Space selected punctuation
	s = rePunctSpace.ReplaceAllString(s, " $1 ")

	// Apostrophe is handled by contractions; keep as-is here.

	// Periods / question / exclamation at word boundaries
	s = reTermPunct.ReplaceAllStringFunc(s, func(m string) string {
		// Put a space before the punctuation and keep following whitespace if present
		// Cases captured: either non-alnum before .!? or alnum before .!? followed by space/eol
		// We'll normalize into " <punct> " and let trimming collapse spaces.
		last := m[len(m)-1:]
		if last == "." || last == "!" || last == "?" {
			return strings.TrimSpace(m[:len(m)-1]) + " " + last + " "
		}
		return m
	})

	// Ellipsis AFTER terminal punctuation processing to avoid interference
	s = reEllipsis.ReplaceAllString(s, " ... ")

	// Commas: space them unless between digits (we will fix numbers afterwards anyway)
	s = strings.ReplaceAll(s, ",", " , ")

	// Handle single quotes that are not part of contractions
	// This regex handles quotes that surround words (not contractions)
	reQuotedWord := regexp.MustCompile(`(\W|^)'(\w+)'(\W|$)`)
	s = reQuotedWord.ReplaceAllString(s, "$1 ' $2 ' $3")

	// Hyphens: keep hyphenated tokens if keepHyphens=true
	if keepHyphens {
		// Do nothing for single '-' between word chars.
		// But make sure standalone hyphens surrounded by spaces are spaced (already will be by user text).
	} else {
		// If not keeping hyphens, space them globally
		s = strings.ReplaceAll(s, "-", " - ")
	}

	// Fix numeric patterns re-joining 1 , 234 and 3 . 14
	s = reNumCommaFix.ReplaceAllString(s, "$1,$2")
	s = reNumDotFix.ReplaceAllString(s, "$1.$2")

	return s
}

func digitsToZero(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if r >= '0' && r <= '9' {
			b.WriteByte('0')
		} else {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func toLowerIfNeeded(s string, lower bool) string {
	if !lower {
		return s
	}
	return strings.ToLower(s)
}

func collapseSpaces(s string) string {
	// Replace any run of whitespace with a single space
	space := false
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if r == '\t' || r == '\n' || r == '\r' || r == ' ' || r == '\v' || r == '\f' {
			if !space {
				b.WriteByte(' ')
				space = true
			}
		} else {
			b.WriteRune(r)
			space = false
		}
	}
	return strings.TrimSpace(b.String())
}
