// ai_wrapper/ai_wrapper.go

package ai_wrapper

import (
	"ProductSetter/bedrock_utils"
	"ProductSetter/openai_utils"
)

const (
	BedrockService = 1
	OpenAIService  = 2
)

// GenerateTitleAndCatchyPhrase is a wrapper function that selects either Bedrock or OpenAI based on serviceType.
func GenerateTitleAndCatchyPhrase(aggregatedText string, retries int, serviceType int) (string, string) {
	switch serviceType {
	case BedrockService:
		// Call Bedrock's title and phrase generator
		return bedrock_utils.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
	case OpenAIService:
		// Call OpenAI's title and phrase generator
		return openai_utils.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
	default:
		// Return default message if serviceType is unknown
		return "No Title", "No Catchy Phrase"
	}
}
