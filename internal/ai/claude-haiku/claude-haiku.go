package claude_haiku

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"imageclust/internal/utils"
)

// Backoff configuration
const (
	initialBackoff = 1 * time.Second
	maxBackoff     = 30 * time.Second
	backoffFactor  = 2.0
	jitterFactor   = 0.3 // Add up to 30% jitter
)

// calculateBackoff returns the backoff duration for a given attempt using exponential backoff with jitter
func calculateBackoff(attempt int) time.Duration {
	backoff := float64(initialBackoff) * math.Pow(backoffFactor, float64(attempt))
	if backoff > float64(maxBackoff) {
		backoff = float64(maxBackoff)
	}
	// Add jitter to prevent thundering herd
	jitter := backoff * jitterFactor * rand.Float64()
	return time.Duration(backoff + jitter)
}

// Claude3Request represents the structure expected by Claude 3
type Claude3Request struct {
	AnthropicVersion string    `json:"anthropic_version"`
	Messages         []Message `json:"messages"`
	MaxTokens        int       `json:"max_tokens"`
	Temperature      float32   `json:"temperature"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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

// InstantiateBedrockClient returns a new instance of BedrockClient
func InstantiateBedrockClient() (*BedrockClient, error) {
	cfg, err := config.LoadDefaultConfig(context.Background(),
		config.WithRegion("us-west-2"),
	)
	if err != nil {
		return nil, fmt.Errorf("unable to load AWS config: %v", err)
	}

	client := bedrockruntime.NewFromConfig(cfg)
	return &BedrockClient{client: client}, nil
}

// GenerateTitleAndCatchyPhrase generates a title and a catchy phrase using Claude via AWS Bedrock
func (b *BedrockClient) GenerateTitleAndCatchyPhrase(aggregatedText string, retries int) (string, string) {
	sanitizedText := utils.TruncateAndSanitize(aggregatedText, 1000)

	for attempt := 0; attempt < retries; attempt++ {
		// Create the request body using the Messages format
		requestBody := Claude3Request{
			AnthropicVersion: "bedrock-2023-05-31",
			Messages: []Message{
				{
					Role: "user",
					Content: fmt.Sprintf(`You are an assistant that generates concise and creative titles and catchy phrases for image clusters.
Each title must be no more than 25 characters, and each catchy phrase must be no more than 100 characters. 
Return the results in JSON format with the fields 'title' and 'catchy_phrase' only.
Do not include any extra text, markdown, or code block formatting in your response.
Ensure that only the JSON object is returned.

Features: %s.`, sanitizedText),
				},
			},
			MaxTokens:   100,
			Temperature: 0.7,
		}

		// Marshal the request body
		requestData, err := json.Marshal(requestBody)
		if err != nil {
			log.Printf("Error marshaling request body: %v", err)
			continue
		}

		// Log the request being sent to Claude
		log.Println("Sending request to Claude 3.5 Haiku via Bedrock:")
		log.Println(string(requestData))

		// Create the Bedrock invoke request
		input := &bedrockruntime.InvokeModelInput{
			ModelId:     aws.String("anthropic.claude-3-haiku-20240307-v1:0"),
			Body:        requestData,
			ContentType: aws.String("application/json"),
			Accept:      aws.String("application/json"),
		}

		// Invoke the model
		output, err := b.client.InvokeModel(context.Background(), input)
		if err != nil {
			backoff := calculateBackoff(attempt)
			log.Printf("Error invoking Bedrock model (attempt %d/%d), retrying in %v: %v", attempt+1, retries, backoff, err)
			time.Sleep(backoff)
			continue
		}

		// Parse the response
		var claudeResp Claude3Response
		err = json.Unmarshal(output.Body, &claudeResp)
		if err != nil {
			backoff := calculateBackoff(attempt)
			log.Printf("Error unmarshaling Claude response (attempt %d/%d), retrying in %v: %v", attempt+1, retries, backoff, err)
			time.Sleep(backoff)
			continue
		}

		// Make sure we have content in the response
		if len(claudeResp.Content) == 0 {
			backoff := calculateBackoff(attempt)
			log.Printf("Empty response from Claude (attempt %d/%d), retrying in %v", attempt+1, retries, backoff)
			time.Sleep(backoff)
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
			backoff := calculateBackoff(attempt)
			log.Printf("Error unmarshaling response JSON (attempt %d/%d), retrying in %v: %v", attempt+1, retries, backoff, err)
			time.Sleep(backoff)
			continue
		}

		// Extract title and catchy_phrase from the response
		title, okTitle := result["title"]
		catchyPhrase, okPhrase := result["catchy_phrase"]
		if !okTitle || !okPhrase {
			backoff := calculateBackoff(attempt)
			log.Printf("Claude response missing 'title' or 'catchy_phrase' (attempt %d/%d), retrying in %v", attempt+1, retries, backoff)
			time.Sleep(backoff)
			continue
		}

		return title, catchyPhrase
	}

	log.Printf("Failed to generate title and catchy phrase after %d retries", retries)
	return "No Title", "No phrase available"
}

// GenerateTitleAndCatchyPhrase is a package-level function that creates a new BedrockClient and calls its method
func GenerateTitleAndCatchyPhrase(aggregatedText string, retries int) (string, string) {
	client, err := InstantiateBedrockClient()
	if err != nil {
		log.Printf("Error creating Bedrock client: %v", err)
		return "No Title", "No phrase available"
	}
	return client.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
}
