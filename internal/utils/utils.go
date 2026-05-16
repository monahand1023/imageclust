package utils

import (
	"strings"
	"unicode/utf8"
)

// SanitizeFilename strips characters unsafe for filesystem use.
func SanitizeFilename(name string) string {
	return strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') ||
			(r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') ||
			r == '.' || r == '-' {
			return r
		}
		return '_'
	}, name)
}

// TruncateAndSanitize truncates input to maxLen runes and removes characters
// that could cause problems in LLM prompts.
func TruncateAndSanitize(input string, maxLen int) string {
	if utf8.RuneCountInString(input) > maxLen {
		truncated := []rune(input)[:maxLen]
		input = string(truncated)
	}
	input = strings.ReplaceAll(input, "\"", "")
	input = strings.ReplaceAll(input, "\\", "")
	input = strings.ReplaceAll(input, "\n", " ")
	input = strings.ReplaceAll(input, "\t", " ")
	input = strings.ReplaceAll(input, "#", "")
	input = strings.ReplaceAll(input, "&", "and")
	input = strings.ReplaceAll(input, "'", "")
	return strings.TrimSpace(input)
}
