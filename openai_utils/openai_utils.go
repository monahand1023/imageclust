// openai_utils/openai_utils.go
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

// ClusterDetails represents the details of a single cluster.
// Ensure this matches the definition in handlers.go
type ClusterDetails struct {
	Title               string
	CatchyPhrase        string
	Labels              string
	Images              []string
	ProductReferenceIDs []string
}

// GenerateTitleAndCatchyPhrase generates a title and a catchy phrase using OpenAI's GPT model.
func GenerateTitleAndCatchyPhrase(aggregatedText string, retries int) (string, string) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Println("OPENAI_API_KEY is not set")
		return "No Title", "No phrase available"
	}

	for attempt := 0; attempt < retries; attempt++ {
		requestBody := map[string]interface{}{
			"model": "gpt-3.5-turbo", // or "gpt-4" if you have access
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
					"content": fmt.Sprintf("Labels: %s.", aggregatedText),
				},
			},
		}

		requestData, err := json.Marshal(requestBody)
		if err != nil {
			log.Printf("Error marshaling OpenAI request body: %v", err)
			continue
		}

		req, err := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(requestData))
		if err != nil {
			log.Printf("Error creating OpenAI request: %v", err)
			continue
		}

		req.Header.Set("Authorization", "Bearer "+apiKey)
		req.Header.Set("Content-Type", "application/json")

		client := &http.Client{
			Timeout: 60 * time.Second, // Increased timeout for API response
		}
		resp, err := client.Do(req)
		if err != nil {
			log.Printf("Error performing OpenAI request: %v", err)
			time.Sleep(2 * time.Second) // Exponential backoff can be implemented
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

		var response struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		err = json.NewDecoder(resp.Body).Decode(&response)
		resp.Body.Close()
		if err != nil {
			log.Printf("Error decoding OpenAI response: %v", err)
			continue
		}

		if len(response.Choices) == 0 {
			log.Println("No choices returned from OpenAI")
			continue
		}

		assistantReply := response.Choices[0].Message.Content

		var result map[string]string
		err = json.Unmarshal([]byte(assistantReply), &result)
		if err != nil {
			log.Printf("Error unmarshaling OpenAI response JSON: %v", err)
			continue
		}

		title, okTitle := result["title"]
		catchyPhrase, okPhrase := result["catchy_phrase"]
		if !okTitle || !okPhrase {
			log.Println("OpenAI response missing 'title' or 'catchy_phrase'")
			continue
		}

		return title, catchyPhrase
	}

	// If all retries fail, return default values
	log.Println("Failed to generate title and catchy phrase after retries")
	return "No Title", "No phrase available"
}
