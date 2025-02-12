package amazon_nova

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

// AmazonNovaMicroResponse represents the structure of the response from Amazon Bedrock
type AmazonNovaMicroResponse struct {
	Results []struct {
		OutputText string `json:"outputText"`
	} `json:"Results"`
}

func GenerateTitleAndCatchyPhrase(aggregatedText string, retries int) (string, string) {
	// Load AWS configuration with explicit region
	cfg, err := config.LoadDefaultConfig(context.TODO(),
		config.WithRegion("us-west-2"),
	)
	if err != nil {
		log.Printf("Unable to load AWS SDK config: %v", err)
		return "No Title", "No phrase available"
	}

	// Create Bedrock client
	client := bedrockruntime.NewFromConfig(cfg)

	// Define the Bedrock model ID you want to use
	modelID := "arn:aws:bedrock:us-west-2:224418580241:inference-profile/us.amazon.nova-micro-v1:0"

	// Truncate and sanitize aggregatedText
	sanitizedText := truncateAndSanitize(aggregatedText, 1000)

	// Construct the prompt text
	promptText := fmt.Sprintf(
		"You are an assistant that generates a single concise and creative title and a catchy phrase for an image cluster. "+
			"The title must be no more than 25 characters, and the catchy phrase must be no more than 100 characters. "+
			"Return the results in JSON format with the fields 'title' and 'catchy_phrase' only. "+
			"Do not include any Markdown or code block formatting in your response. "+
			"Ensure that only one JSON object is returned, containing only these two fields. "+
			"Features: %s.",
		sanitizedText,
	)

	// Create the request payload as a map
	requestPayload := map[string]string{
		"inputText": promptText,
	}

	// Marshal the request payload to JSON
	requestBody, err := json.Marshal(requestPayload)
	if err != nil {
		log.Printf("Error marshaling request body: %v", err)
		return "No Title", "No phrase available"
	}

	for attempt := 0; attempt < retries; attempt++ {
		// Create the request input
		reqInput := &bedrockruntime.InvokeModelInput{
			ModelId:     aws.String(modelID),
			Body:        requestBody,
			ContentType: aws.String("application/json"),
			Accept:      aws.String("application/json"),
		}

		// Log the request being sent to Bedrock
		log.Println("Sending request to Amazon Bedrock:")
		log.Println(string(requestBody))

		// Send the request to Bedrock
		resp, err := client.InvokeModel(context.TODO(), reqInput)
		if err != nil {
			log.Printf("Error invoking Bedrock model: %v", err)
			time.Sleep(2 * time.Second)
			continue
		}

		// Handle response as []byte
		bodyBytes := resp.Body

		// Parse the response JSON
		var bedrockResp AmazonNovaMicroResponse
		err = json.Unmarshal(bodyBytes, &bedrockResp)
		if err != nil {
			log.Printf("Error unmarshaling Bedrock response: %v", err)
			time.Sleep(2 * time.Second)
			continue
		}

		// Check if any results are returned
		if len(bedrockResp.Results) == 0 {
			log.Println("No results returned from Bedrock")
			time.Sleep(2 * time.Second)
			continue
		}

		assistantReply := bedrockResp.Results[0].OutputText

		// Log the response received from Bedrock
		log.Println("Received response from Amazon Bedrock:")
		log.Println(assistantReply)

		// Attempt to unmarshal the assistant's reply into a map
		var result map[string]interface{}
		err = json.Unmarshal([]byte(assistantReply), &result)
		if err != nil {
			log.Printf("Error unmarshaling Bedrock response JSON: %v", err)
			time.Sleep(2 * time.Second)
			continue
		}

		// Function to extract a string from the result
		extractString := func(value interface{}) (string, bool) {
			switch v := value.(type) {
			case string:
				return v, true
			case []interface{}:
				if len(v) > 0 {
					if str, ok := v[0].(string); ok {
						return str, true
					}
				}
			}
			return "", false
		}

		// Extract title
		titleValue, okTitle := result["title"]
		title, okTitleExtracted := extractString(titleValue)

		// Extract catchy_phrase
		catchyPhraseValue, okPhrase := result["catchy_phrase"]
		catchyPhrase, okPhraseExtracted := extractString(catchyPhraseValue)

		if !okTitle || !okTitleExtracted || !okPhrase || !okPhraseExtracted {
			log.Println("Bedrock response missing 'title' or 'catchy_phrase'")
			time.Sleep(2 * time.Second)
			continue
		}

		return title, catchyPhrase
	}

	// If all retries fail, return default values
	log.Println("Failed to generate title and catchy phrase after retries")
	return "No Title", "No phrase available"
}

// truncateAndSanitize truncates the input string to a maximum length and removes or replaces characters that could interfere with JSON formatting.
func truncateAndSanitize(input string, maxLen int) string {
	// Truncate the string to the maximum length, ensuring we don't cut in the middle of a multi-byte character
	if utf8.RuneCountInString(input) > maxLen {
		truncated := []rune(input)[:maxLen]
		input = string(truncated)
	}

	// Replace or remove problematic characters
	input = strings.ReplaceAll(input, "\"", "")   // Remove double quotes to prevent JSON issues
	input = strings.ReplaceAll(input, "\\", "")   // Remove backslashes
	input = strings.ReplaceAll(input, "\n", " ")  // Replace newlines with space
	input = strings.ReplaceAll(input, "\t", " ")  // Replace tabs with space
	input = strings.ReplaceAll(input, "#", "")    // Remove hash symbols
	input = strings.ReplaceAll(input, "&", "and") // Replace ampersands with 'and'
	input = strings.ReplaceAll(input, "'", "")    // Remove single quotes if necessary
	input = strings.TrimSpace(input)              // Remove leading and trailing spaces

	return input
}
