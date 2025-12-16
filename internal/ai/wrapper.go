package ai

import (
	"sort"
	"sync"

	"imageclust/internal/ai/claude-haiku"
)

const (
	ClaudeHaikuService = 1
)

// ServiceConfig represents a service configuration
type ServiceConfig struct {
	ServiceType int
	Name        string
	Model       interface{} // Can hold OpenAIModel or other model configs
	Order       int         // Added to control display order
}

// ModelOutput represents the output from a single model
type ModelOutput struct {
	ServiceName  string
	Title        string
	CatchyPhrase string
	Order        int // Added to control display order
}

// AvailableServices defines all available AI services in desired order
var AvailableServices = []ServiceConfig{
	{
		ServiceType: ClaudeHaikuService,
		Name:        "Claude Haiku v3.5",
		Model:       nil,
		Order:       1,
	},
}

// GenerateTitleAndCatchyPhrase maintains backward compatibility
func GenerateTitleAndCatchyPhrase(aggregatedText string, retries int, serviceType int) (string, string) {
	switch serviceType {
	case ClaudeHaikuService:
		return claude_haiku.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
	default:
		return "No Title", "No Catchy Phrase"
	}
}

// GenerateTitleAndCatchyPhraseMultiService generates titles and catchy phrases using all available services
func GenerateTitleAndCatchyPhraseMultiService(aggregatedText string, retries int) []ModelOutput {
	outputs := make([]ModelOutput, 0, len(AvailableServices))
	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, service := range AvailableServices {
		wg.Add(1)
		go func(svc ServiceConfig) {
			defer wg.Done()

			var title, catchyPhrase string

			switch svc.ServiceType {
			case ClaudeHaikuService:
				title, catchyPhrase = claude_haiku.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
			}

			mu.Lock()
			outputs = append(outputs, ModelOutput{
				ServiceName:  svc.Name,
				Title:        title,
				CatchyPhrase: catchyPhrase,
				Order:        svc.Order,
			})
			mu.Unlock()
		}(service)
	}

	wg.Wait()

	// Sort outputs by Order
	sort.Slice(outputs, func(i, j int) bool {
		return outputs[i].Order < outputs[j].Order
	})

	return outputs
}
