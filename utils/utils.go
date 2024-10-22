// utils/utils.go
package utils

import (
	"ProductSetter/models"
	"fmt"
	"html/template"
	"os"
	"strings"
)

// GenerateHTMLOutput generates an HTML file representing the clusters.
func GenerateHTMLOutput(clusterDetails map[string]models.ClusterDetails, tempDir, host string, port int) (string, error) {
	htmlPath := fmt.Sprintf("%s/clusters.html", tempDir)

	// Define a simple HTML template
	const tmpl = `
	<!DOCTYPE html>
	<html>
	<head>
		<title>Product Clusters</title>
		<style>
			body { font-family: Arial, sans-serif; }
			.cluster { border: 1px solid #ccc; padding: 10px; margin: 10px; }
			.cluster-title { font-size: 1.5em; font-weight: bold; }
			.cluster-phrase { font-style: italic; }
			.images { display: flex; flex-wrap: wrap; }
			.images img { max-width: 100px; margin: 5px; }
		</style>
	</head>
	<body>
		<h1>Product Clusters</h1>
		{{range $key, $cluster := .}}
			<div class="cluster">
				<div class="cluster-title">{{$cluster.Title}}</div>
				<div class="cluster-phrase">{{$cluster.CatchyPhrase}}</div>
				<div class="labels"><strong>Labels:</strong> {{$cluster.Labels}}</div>
				<div class="images">
					{{range $cluster.Images}}
						<img src="file://{{$.ImageDir}}/{{$}}" alt="Product Image">
					{{end}}
				</div>
			</div>
		{{end}}
	</body>
	</html>
	`

	// Parse the template
	t, err := template.New("clusters").Parse(tmpl)
	if err != nil {
		return "", fmt.Errorf("failed to parse HTML template: %v", err)
	}

	// Create the HTML file
	file, err := os.Create(htmlPath)
	if err != nil {
		return "", fmt.Errorf("failed to create HTML file: %v", err)
	}
	defer file.Close()

	// Execute the template with cluster details
	data := struct {
		Clusters map[string]models.ClusterDetails
		ImageDir string
	}{
		Clusters: clusterDetails,
		ImageDir: tempDir + "/images", // Adjust based on actual image directory
	}

	err = t.Execute(file, data)
	if err != nil {
		return "", fmt.Errorf("failed to execute HTML template: %v", err)
	}

	return htmlPath, nil
}

// sanitizeFilename replaces invalid characters in filenames
func SanitizeFilename(name string) string {
	// Replace any character that is not a letter, number, dot, or dash with an underscore
	return strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') ||
			(r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') ||
			r == '.' || r == '-' {
			return r
		}
		return '_'
	}, name)
}

// urlEncode encodes a string for safe inclusion in URLs
func URLEncode(s string) string {
	return strings.ReplaceAll(s, " ", "%20")
}

// CleanText performs basic text cleaning, such as removing extra spaces and trimming.
func CleanText(text string) string {
	return strings.TrimSpace(text)
}
