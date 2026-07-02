package utils

import (
	"testing"
)

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
