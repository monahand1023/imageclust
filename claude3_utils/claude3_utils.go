package claude3_utils

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// Claude3Request represents the structure expected by Claude 3
type Claude3Request struct {
	AnthropicVersion string `json:"anthropic_version"`
	Messages         []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	MaxTokens int `json:"max_tokens"`
}

// Claude3Response represents the structure of the response from Claude 3
type Claude3Response struct {
	Content []struct {
		Text string `json:"text"`
	} `json:"content"`
}

// BedrockClient implements the AIClient interface using AWS Bedrock's Claude
type BedrockClient struct {
	client *bedrockruntime.Client
}

// NewBedrockClient returns a new instance of BedrockClient
func NewBedrockClient() (*BedrockClient, error) {
	cfg, err := config.LoadDefaultConfig(context.Background(),
		config.WithRegion("us-east-1"),
	)
	if err != nil {
		return nil, fmt.Errorf("unable to load AWS config: %v", err)
	}

	client := bedrockruntime.NewFromConfig(cfg)
	return &BedrockClient{client: client}, nil
}

// GenerateTitleAndCatchyPhrase generates a title and a catchy phrase using Claude via AWS Bedrock
func (b *BedrockClient) GenerateTitleAndCatchyPhrase(aggregatedText string, retries int) (string, string) {
	sanitizedText := truncateAndSanitize(aggregatedText, 1000)

	for attempt := 0; attempt < retries; attempt++ {
		prompt := fmt.Sprintf(`You are an assistant that generates concise and creative titles and catchy phrases for product clusters.
Each title must be no more than 25 characters, and each catchy phrase must be no more than 100 characters. 
Use first-person voice; avoid using 'we' and express using 'I' or 'my'. 
Return the results in JSON format with the fields 'title' and 'catchy_phrase' only.
Do not include any extra text, markdown, or code block formatting in your response.
Ensure that only the JSON object is returned.

Features: %s.`, sanitizedText)

		// Create the request body
		requestBody := Claude3Request{
			AnthropicVersion: "bedrock-2023-05-31",
			Messages: []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			}{
				{
					Role:    "user",
					Content: prompt,
				},
			},
			MaxTokens: 100,
		}

		// Marshal the request body
		requestData, err := json.Marshal(requestBody)
		if err != nil {
			log.Printf("Error marshaling request body: %v", err)
			continue
		}

		// Log the request being sent to Claude
		log.Println("Sending request to Claude 3 via Bedrock:")
		log.Println(string(requestData))

		// Create the Bedrock invoke request
		input := &bedrockruntime.InvokeModelInput{
			ModelId:     aws.String("anthropic.claude-3-5-sonnet-20240620-v1:0"),
			Body:        requestData,
			ContentType: aws.String("application/json"),
			Accept:      aws.String("application/json"),
		}

		// Invoke the model
		output, err := b.client.InvokeModel(context.Background(), input)
		if err != nil {
			log.Printf("Error invoking Bedrock model: %v", err)
			time.Sleep(2 * time.Second)
			continue
		}

		// Parse the response
		var claudeResp Claude3Response
		err = json.Unmarshal(output.Body, &claudeResp)
		if err != nil {
			log.Printf("Error unmarshaling Claude response: %v", err)
			time.Sleep(2 * time.Second)
			continue
		}

		// Make sure we have content in the response
		if len(claudeResp.Content) == 0 {
			log.Println("Empty response from Claude")
			time.Sleep(2 * time.Second)
			continue
		}

		responseText := claudeResp.Content[0].Text

		// Log the response received from Claude
		log.Println("Received response from Claude:")
		log.Println(responseText)

		// Attempt to parse the response as JSON
		var result map[string]string
		err = json.Unmarshal([]byte(responseText), &result)
		if err != nil {
			log.Printf("Error unmarshaling response JSON: %v", err)
			time.Sleep(2 * time.Second)
			continue
		}

		// Extract title and catchy_phrase from the response
		title, okTitle := result["title"]
		catchyPhrase, okPhrase := result["catchy_phrase"]
		if !okTitle || !okPhrase {
			log.Println("Claude response missing 'title' or 'catchy_phrase'")
			time.Sleep(2 * time.Second)
			continue
		}

		return title, catchyPhrase
	}

	log.Println("Failed to generate title and catchy phrase after retries")
	return "No Title", "No phrase available"
}

// truncateAndSanitize truncates the input string to a maximum length and removes or replaces problematic characters
func truncateAndSanitize(input string, maxLen int) string {
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
	input = strings.TrimSpace(input)

	return input
}

// GenerateTitleAndCatchyPhrase is a package-level function that creates a new BedrockClient and calls its method
func GenerateTitleAndCatchyPhrase(aggregatedText string, retries int) (string, string) {
	client, err := NewBedrockClient()
	if err != nil {
		log.Printf("Error creating Bedrock client: %v", err)
		return "No Title", "No phrase available"
	}
	return client.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
}
