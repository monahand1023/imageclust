package ai_wrapper

import (
	"ProductSetter/bedrock_utils"
	"ProductSetter/claude3_utils"
	"ProductSetter/claude_utils"
	"ProductSetter/openai_utils"
	"sync"
)

const (
	BedrockService = 1
	OpenAIService  = 2
	ClaudeService  = 3
	Claude3Service = 4
)

// ServiceConfig represents a service configuration
type ServiceConfig struct {
	ServiceType int
	Name        string
}

// ModelOutput represents the output from a single model
type ModelOutput struct {
	ServiceName  string
	Title        string
	CatchyPhrase string
}

// AvailableServices defines all available AI services
var AvailableServices = []ServiceConfig{
	{ServiceType: BedrockService, Name: "Amazon Titan"},
	{ServiceType: OpenAIService, Name: "GPT-4"},
	{ServiceType: ClaudeService, Name: "Claude 2"},
	{ServiceType: Claude3Service, Name: "Claude 3"},
}

// GenerateTitleAndCatchyPhrase is maintained for backward compatibility
func GenerateTitleAndCatchyPhrase(aggregatedText string, retries int, serviceType int) (string, string) {
	outputs := GenerateTitleAndCatchyPhraseMultiService(aggregatedText, retries)
	// Return the result from the specified service, or default values if not found
	for _, output := range outputs {
		for _, service := range AvailableServices {
			if service.ServiceType == serviceType && service.Name == output.ServiceName {
				return output.Title, output.CatchyPhrase
			}
		}
	}
	return "No Title", "No Catchy Phrase"
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
			case BedrockService:
				title, catchyPhrase = bedrock_utils.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
			case OpenAIService:
				title, catchyPhrase = openai_utils.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
			case ClaudeService:
				title, catchyPhrase = claude_utils.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
			case Claude3Service:
				title, catchyPhrase = claude3_utils.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
			default:
				title, catchyPhrase = "No Title", "No Catchy Phrase"
			}

			mu.Lock()
			outputs = append(outputs, ModelOutput{
				ServiceName:  svc.Name,
				Title:        title,
				CatchyPhrase: catchyPhrase,
			})
			mu.Unlock()
		}(service)
	}

	wg.Wait()
	return outputs
}
