package utils

import (
	"fmt"
	"html"
	"html/template"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// CleanText removes HTML tags, unescapes HTML entities, and trims extra whitespace.
// It ensures that the returned text is clean and suitable for further processing.
func CleanText(text string) string {
	// Remove HTML tags using a regular expression
	re := regexp.MustCompile(`<[^>]+>`)
	text = re.ReplaceAllString(text, "")

	// Unescape HTML entities (e.g., &amp; -> &)
	text = html.UnescapeString(text)

	// Remove extra whitespace by splitting and rejoining the string
	text = strings.Join(strings.Fields(text), " ")

	return text
}

// ConcatenateVectors concatenates multiple slices of float64 into a single slice.
// This is useful for combining embedding vectors or other numerical data.
func ConcatenateVectors(vectors [][]float64) []float64 {
	var result []float64
	for _, vector := range vectors {
		result = append(result, vector...)
	}
	return result
}

// ClusterDetails represents the details of a single cluster.
// It includes the title, catchy phrase, labels, associated images, and product reference IDs.
type ClusterDetails struct {
	Title               string   `json:"title"`
	CatchyPhrase        string   `json:"catchy_phrase"`
	Labels              string   `json:"labels"`
	Images              []string `json:"images"`
	ProductReferenceIDs []string `json:"product_reference_ids"`
}

// GenerateHTMLOutput generates an HTML file displaying the clusters.
// It takes a map of cluster IDs to ClusterDetails, the directory where images are stored,
// the host, and the port where images are served.
// Returns the path to the generated HTML file or an error if any occurs.
func GenerateHTMLOutput(clusters map[string]ClusterDetails, imageDir string, host string, port int) (string, error) {
	// Define the HTML template
	const tpl = `
	<!DOCTYPE html>
	<html>
	<head>
		<title>Product Clusters</title>
		<style>
			body { font-family: Arial, sans-serif; }
			.cluster { border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
			.cluster-title { font-size: 1.5em; margin-bottom: 5px; }
			.cluster-phrase { font-style: italic; margin-bottom: 5px; }
			.cluster-labels { color: #555; margin-bottom: 10px; }
			.cluster-images img { max-width: 150px; margin-right: 10px; }
		</style>
	</head>
	<body>
		<h1>Product Clusters</h1>
		{{range $id, $cluster := .Clusters}}
			<div class="cluster">
				<div class="cluster-title">{{$cluster.Title}}</div>
				<div class="cluster-phrase">{{$cluster.CatchyPhrase}}</div>
				<div class="cluster-labels">Labels: {{$cluster.Labels}}</div>
				<div class="cluster-images">
					{{range $cluster.Images}}
						<img src="{{$.ImageBaseURL}}/{{$ .}}" alt="Product Image">
					{{end}}
				</div>
			</div>
		{{end}}
	</body>
	</html>
	`

	// Parse the template
	t, err := template.New("clusters").Parse(tpl)
	if err != nil {
		return "", fmt.Errorf("error parsing HTML template: %v", err)
	}

	// Prepare data for the template
	type TemplateData struct {
		Clusters     map[string]ClusterDetails
		ImageBaseURL string
	}

	data := TemplateData{
		Clusters:     clusters,
		ImageBaseURL: fmt.Sprintf("http://%s:%d", host, port),
	}

	// Define the output directory and ensure it exists
	outputDir := "output_html"
	err = os.MkdirAll(outputDir, 0755)
	if err != nil {
		return "", fmt.Errorf("error creating output directory: %v", err)
	}

	// Define the output file path
	outputPath := filepath.Join(outputDir, "clusters.html")

	// Create the output file
	file, err := os.Create(outputPath)
	if err != nil {
		return "", fmt.Errorf("error creating HTML file: %v", err)
	}
	defer file.Close()

	// Execute the template and write to the file
	err = t.Execute(file, data)
	if err != nil {
		return "", fmt.Errorf("error executing HTML template: %v", err)
	}

	return outputPath, nil
}
