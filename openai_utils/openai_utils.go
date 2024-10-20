package openai_utils

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
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

		requestData, _ := json.Marshal(requestBody)

		req, err := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(requestData))
		if err != nil {
			continue
		}

		req.Header.Set("Authorization", "Bearer "+apiKey)
		req.Header.Set("Content-Type", "application/json")

		client := &http.Client{}
		resp, err := client.Do(req)
		if err != nil {
			continue
		}
		defer resp.Body.Close()

		var response struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
			continue
		}

		if len(response.Choices) == 0 {
			continue
		}

		assistantReply := response.Choices[0].Message.Content

		var result map[string]string
		if err := json.Unmarshal([]byte(assistantReply), &result); err != nil {
			continue
		}

		title, okTitle := result["title"]
		catchyPhrase, okPhrase := result["catchy_phrase"]
		if !okTitle || !okPhrase {
			continue
		}

		return title, catchyPhrase
	}

	return "No Title", "No phrase available"
}
