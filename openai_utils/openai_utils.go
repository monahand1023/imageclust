package openai_utils

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"
)

// OpenAIModel represents a specific OpenAI model configuration
type OpenAIModel struct {
	ModelName   string
	ServiceName string
}

// Available OpenAI models
var (
	GPT4 = OpenAIModel{
		ModelName:   "gpt-4",
		ServiceName: "GPT-4",
	}
	GPT35Turbo = OpenAIModel{
		ModelName:   "gpt-3.5-turbo",
		ServiceName: "GPT-3.5 Turbo",
	}
)

// GPTResponse represents the structure of the response from OpenAI
type GPTResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// OpenAIClient implements the AIClient interface using OpenAI's GPT
type OpenAIClient struct {
	Model OpenAIModel
}

// NewOpenAIClient returns a new instance of OpenAIClient
func NewOpenAIClient(model OpenAIModel) *OpenAIClient {
	return &OpenAIClient{
		Model: model,
	}
}

// GenerateTitleAndCatchyPhrase generates a title and a catchy phrase using OpenAI's GPT model
func (o *OpenAIClient) GenerateTitleAndCatchyPhrase(aggregatedText string, retries int) (string, string) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Println("OPENAI_API_KEY is not set")
		return "No Title", "No phrase available"
	}

	for attempt := 0; attempt < retries; attempt++ {
		// Construct the request body
		requestBody := map[string]interface{}{
			"model": o.Model.ModelName,
			"messages": []map[string]string{
				{
					"role": "system",
					"content": "You are an assistant that generates concise and creative titles and catchy phrases for product clusters. " +
						"Each title must be no more than 25 characters, and each catchy phrase must be no more than 100 characters. " +
						"Use first-person voice; avoid using 'we' and express using 'I' or 'my'. " +
						"Return the results in JSON format with the fields 'title' and 'catchy_phrase' only. " +
						"Do not include any Markdown or code block formatting in your response. " +
						"Ensure that only one JSON object is returned.",
				},
				{
					"role":    "user",
					"content": fmt.Sprintf("Features: %s.", aggregatedText),
				},
			},
		}

		// Marshal the request body to JSON
		requestData, err := json.Marshal(requestBody)
		if err != nil {
			log.Printf("Error marshaling OpenAI request body: %v", err)
			continue
		}

		// Log the request being sent to GPT
		log.Printf("Sending request to OpenAI (%s):", o.Model.ServiceName)
		var prettyRequest bytes.Buffer
		err = json.Indent(&prettyRequest, requestData, "", "  ")
		if err != nil {
			log.Printf("Error formatting request JSON: %v", err)
			log.Println(string(requestData)) // Fallback to raw JSON
		} else {
			log.Println(prettyRequest.String())
		}

		// Create the HTTP POST request
		req, err := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(requestData))
		if err != nil {
			log.Printf("Error creating OpenAI request: %v", err)
			continue
		}

		// Set the necessary headers
		req.Header.Set("Authorization", "Bearer "+apiKey)
		req.Header.Set("Content-Type", "application/json")

		// Initialize the HTTP client with a timeout
		client := &http.Client{
			Timeout: 60 * time.Second, // Increased timeout for API response
		}

		// Send the request to OpenAI
		resp, err := client.Do(req)
		if err != nil {
			log.Printf("Error performing OpenAI request: %v", err)
			time.Sleep(2 * time.Second) // Simple backoff strategy
			continue
		}

		// Handle rate limiting or server errors
		if resp.StatusCode == http.StatusTooManyRequests {
			log.Printf("OpenAI rate limit exceeded. Attempt %d/%d", attempt+1, retries)
			resp.Body.Close()
			time.Sleep(2 * time.Second)
			continue
		} else if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			log.Printf("OpenAI API error. Status: %d, Response: %s", resp.StatusCode, string(bodyBytes))
			resp.Body.Close()
			time.Sleep(2 * time.Second)
			continue
		}

		// Read and decode the response
		var gptResp GPTResponse
		err = json.NewDecoder(resp.Body).Decode(&gptResp)
		resp.Body.Close()
		if err != nil {
			log.Printf("Error decoding OpenAI response: %v", err)
			continue
		}

		// Check if any choices are returned
		if len(gptResp.Choices) == 0 {
			log.Println("No choices returned from OpenAI")
			continue
		}

		assistantReply := gptResp.Choices[0].Message.Content

		// Log the response received from GPT
		log.Printf("Received response from OpenAI (%s):", o.Model.ServiceName)
		log.Println(assistantReply)

		// Attempt to unmarshal the JSON response
		var result map[string]string
		err = json.Unmarshal([]byte(assistantReply), &result)
		if err != nil {
			log.Printf("Error unmarshaling OpenAI response JSON: %v", err)
			continue
		}

		// Extract title and catchy_phrase from the response
		title, okTitle := result["title"]
		catchyPhrase, okPhrase := result["catchy_phrase"]
		if !okTitle || !okPhrase {
			log.Println("OpenAI response missing 'title' or 'catchy_phrase'")
			continue
		}

		return title, catchyPhrase
	}

	// If all retries fail, return default values
	log.Printf("Failed to generate title and catchy phrase after %d retries using %s", retries, o.Model.ServiceName)
	return "No Title", "No phrase available"
}

// GenerateTitleAndCatchyPhrase is a package-level function that creates a new OpenAIClient and calls its method
func GenerateTitleAndCatchyPhrase(aggregatedText string, retries int, model OpenAIModel) (string, string) {
	client := NewOpenAIClient(model)
	return client.GenerateTitleAndCatchyPhrase(aggregatedText, retries)
}
