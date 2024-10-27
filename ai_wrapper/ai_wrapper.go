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
	GPT4Service    = 2
	GPT35Service   = 3
	ClaudeService  = 4
	Claude3Service = 5
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
		ServiceType: BedrockService,
		Name:        "Amazon Titan Text G1 - Premier",
		Model:       nil,
		Order:       1,
	},
	{
		ServiceType: GPT35Service,
		Name:        "OpenAI GPT-3.5 Turbo",
		Model:       openai_utils.GPT35Turbo,
		Order:       2,
	},
	{
		ServiceType: GPT4Service,
		Name:        "Open AI GPT-4",
		Model:       openai_utils.GPT4,
		Order:       3,
	},
	{
		ServiceType: ClaudeService,
		Name:        "Claude 2",
		Model:       nil,
		Order:       4,
	},
	{
		ServiceType: Claude3Service,
		Name:        "Claude 3.5 Sonnet",
		Model:       nil,
		Order:       5,
	},
}

// GenerateTitleAndCatchyPhrase maintains backward compatibility
func GenerateTitleAndCatchyPhrase(aggregatedText string, retries int, serviceType int) (string, string) {
	switch serviceType {
	case BedrockService:
		return bedrock_utils.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
	case GPT4Service:
		return openai_utils.GenerateTitleAndCatchyPhrase(aggregatedText, retries, openai_utils.GPT4)
	case GPT35Service:
		return openai_utils.GenerateTitleAndCatchyPhrase(aggregatedText, retries, openai_utils.GPT35Turbo)
	case ClaudeService:
		return claude_utils.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
	case Claude3Service:
		return claude3_utils.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
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
			case BedrockService:
				title, catchyPhrase = bedrock_utils.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
			case GPT4Service, GPT35Service:
				if openaiModel, ok := svc.Model.(openai_utils.OpenAIModel); ok {
					title, catchyPhrase = openai_utils.GenerateTitleAndCatchyPhrase(aggregatedText, retries, openaiModel)
				}
			case ClaudeService:
				title, catchyPhrase = claude_utils.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
			case Claude3Service:
				title, catchyPhrase = claude3_utils.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
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

	// Sort outputs by Order before returning
	sortedOutputs := make([]ModelOutput, len(outputs))
	copy(sortedOutputs, outputs)
	for i := 0; i < len(sortedOutputs)-1; i++ {
		for j := i + 1; j < len(sortedOutputs); j++ {
			if sortedOutputs[i].Order > sortedOutputs[j].Order {
				sortedOutputs[i], sortedOutputs[j] = sortedOutputs[j], sortedOutputs[i]
			}
		}
	}

	return sortedOutputs
}
