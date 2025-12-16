package claude_haiku

import (
	"testing"
	"time"
)

func TestCalculateBackoff(t *testing.T) {
	tests := []struct {
		name        string
		attempt     int
		minExpected time.Duration
		maxExpected time.Duration
	}{
		{
			name:        "first attempt",
			attempt:     0,
			minExpected: 1 * time.Second,
			maxExpected: 1300 * time.Millisecond, // 1s + 30% jitter
		},
		{
			name:        "second attempt",
			attempt:     1,
			minExpected: 2 * time.Second,
			maxExpected: 2600 * time.Millisecond, // 2s + 30% jitter
		},
		{
			name:        "third attempt",
			attempt:     2,
			minExpected: 4 * time.Second,
			maxExpected: 5200 * time.Millisecond, // 4s + 30% jitter
		},
		{
			name:        "high attempt (should cap at maxBackoff)",
			attempt:     10,
			minExpected: 30 * time.Second,
			maxExpected: 39 * time.Second, // 30s + 30% jitter
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Run multiple times to account for jitter
			for i := 0; i < 10; i++ {
				result := calculateBackoff(tt.attempt)
				if result < tt.minExpected {
					t.Errorf("calculateBackoff(%d) = %v, want >= %v", tt.attempt, result, tt.minExpected)
				}
				if result > tt.maxExpected {
					t.Errorf("calculateBackoff(%d) = %v, want <= %v", tt.attempt, result, tt.maxExpected)
				}
			}
		})
	}
}

func TestCalculateBackoff_Increasing(t *testing.T) {
	// Verify backoff generally increases with each attempt (ignoring jitter)
	prev := time.Duration(0)
	for attempt := 0; attempt < 5; attempt++ {
		// Get average of multiple samples to account for jitter
		var total time.Duration
		samples := 10
		for i := 0; i < samples; i++ {
			total += calculateBackoff(attempt)
		}
		avg := total / time.Duration(samples)

		if attempt > 0 && avg <= prev {
			t.Errorf("backoff should increase: attempt %d avg=%v <= previous avg=%v", attempt, avg, prev)
		}
		prev = avg
	}
}

func TestCalculateBackoff_MaxCap(t *testing.T) {
	// Verify backoff is capped at maxBackoff
	for attempt := 10; attempt < 20; attempt++ {
		result := calculateBackoff(attempt)
		// With 30% jitter, max should be 30s * 1.3 = 39s
		if result > 39*time.Second {
			t.Errorf("calculateBackoff(%d) = %v, should not exceed maxBackoff + jitter", attempt, result)
		}
	}
}
