package utils

import (
	"strings"
	"testing"
)

func TestTruncateAndSanitize_Truncation(t *testing.T) {
	input := "This is a very long string that should be truncated"
	result := TruncateAndSanitize(input, 10)

	if len([]rune(result)) > 10 {
		t.Errorf("expected max 10 runes, got %d", len([]rune(result)))
	}
}

func TestTruncateAndSanitize_NoTruncationNeeded(t *testing.T) {
	input := "short"
	result := TruncateAndSanitize(input, 100)

	if result != "short" {
		t.Errorf("expected 'short', got '%s'", result)
	}
}

func TestTruncateAndSanitize_RemovesQuotes(t *testing.T) {
	input := `He said "hello"`
	result := TruncateAndSanitize(input, 100)

	if strings.Contains(result, `"`) {
		t.Error("expected quotes to be removed")
	}
}

func TestTruncateAndSanitize_RemovesBackslash(t *testing.T) {
	input := `path\to\file`
	result := TruncateAndSanitize(input, 100)

	if strings.Contains(result, `\`) {
		t.Error("expected backslashes to be removed")
	}
}

func TestTruncateAndSanitize_ReplacesNewlines(t *testing.T) {
	input := "line1\nline2\nline3"
	result := TruncateAndSanitize(input, 100)

	if strings.Contains(result, "\n") {
		t.Error("expected newlines to be replaced with spaces")
	}
}

func TestTruncateAndSanitize_ReplacesTabs(t *testing.T) {
	input := "col1\tcol2\tcol3"
	result := TruncateAndSanitize(input, 100)

	if strings.Contains(result, "\t") {
		t.Error("expected tabs to be replaced with spaces")
	}
}

func TestTruncateAndSanitize_RemovesHash(t *testing.T) {
	input := "#hashtag and #another"
	result := TruncateAndSanitize(input, 100)

	if strings.Contains(result, "#") {
		t.Error("expected hash symbols to be removed")
	}
}

func TestTruncateAndSanitize_ReplacesAmpersand(t *testing.T) {
	input := "fish & chips"
	result := TruncateAndSanitize(input, 100)

	if strings.Contains(result, "&") {
		t.Error("expected ampersand to be replaced with 'and'")
	}
	if !strings.Contains(result, "and") {
		t.Error("expected 'and' to replace '&'")
	}
}

func TestTruncateAndSanitize_RemovesSingleQuotes(t *testing.T) {
	input := "it's a test"
	result := TruncateAndSanitize(input, 100)

	if strings.Contains(result, "'") {
		t.Error("expected single quotes to be removed")
	}
}

func TestTruncateAndSanitize_TrimsWhitespace(t *testing.T) {
	input := "   hello world   "
	result := TruncateAndSanitize(input, 100)

	if result != "hello world" {
		t.Errorf("expected 'hello world', got '%s'", result)
	}
}

func TestTruncateAndSanitize_Unicode(t *testing.T) {
	// Unicode characters should be handled correctly
	input := "Hello 世界 🌍"
	result := TruncateAndSanitize(input, 5)

	// Should truncate to 5 runes, not bytes
	if len([]rune(result)) > 5 {
		t.Errorf("expected max 5 runes for unicode, got %d", len([]rune(result)))
	}
}

func TestTruncateAndSanitize_EmptyString(t *testing.T) {
	input := ""
	result := TruncateAndSanitize(input, 100)

	if result != "" {
		t.Errorf("expected empty string, got '%s'", result)
	}
}

func TestTruncateAndSanitize_AllSpecialChars(t *testing.T) {
	input := `"'\#&`
	result := TruncateAndSanitize(input, 100)

	// Should only contain "and" from the ampersand replacement
	expected := "and"
	if result != expected {
		t.Errorf("expected '%s', got '%s'", expected, result)
	}
}

func TestSanitizeFilename(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"normal-file.jpg", "normal-file.jpg"},
		{"file with spaces.png", "file_with_spaces.png"},
		{"file@special#chars.gif", "file_special_chars.gif"},
		{"UPPERCASE.JPG", "UPPERCASE.JPG"},
		{"mixed-Case_123.webp", "mixed-Case_123.webp"},
		{"", ""},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := SanitizeFilename(tt.input)
			if result != tt.expected {
				t.Errorf("SanitizeFilename(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}
