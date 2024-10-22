// utils/utils.go
package utils

import (
	"ProductSetter/models"
	"bytes"
	"fmt"
	"html/template"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// GenerateHTMLOutput generates an HTML file based on cluster details.
func GenerateHTMLOutput(clusters map[string]models.ClusterDetails, tempDir string) (string, error) {
	const tmpl = `
<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<title>Clustered Fashion Items</title>
	<style>
		.container {
			width: 80%;
			margin: auto;
		}
		.cluster {
			border: 1px solid #ccc;
			padding: 20px;
			margin-bottom: 20px;
		}
		.image-container {
			display: flex;
			flex-wrap: wrap;
		}
		.image {
			margin: 10px;
		}
		.image img {
			max-width: 200px;
			height: auto;
		}
		.submit-button {
			background-color: #4CAF50;
			color: white;
			padding: 10px 20px;
			border: none;
			cursor: pointer;
		}
	</style>
	<script>
		async function publishCluster(clusterId, title, catchyPhrase, productReferenceIds) {
			const payload = {
				cluster_id: clusterId,
				title: title,
				description: catchyPhrase, // Using catchyPhrase instead of description
				product_reference_ids: productReferenceIds
			};

			try {
				const response = await fetch("/publish", {
					method: "POST",
					headers: {
						"Content-Type": "application/json"
					},
					body: JSON.stringify(payload)
				});

				const data = await response.json();
				if (data.success) {
					alert("Cluster published successfully!");
				} else {
					alert("Failed to publish cluster: " + data.error);
				}
			} catch (error) {
				console.error("Error:", error);
				alert("An error occurred while publishing the cluster.");
			}
		}
	</script>
</head>
<body>
	<div class="container">
		<h1>Clustered Fashion Items</h1>
		<!-- Clusters -->
		{{range $cluster_id, $cluster_info := .Clusters}}
			<div class="cluster">
				<h2>{{ $cluster_info.Title }}</h2>
				<p><strong>Labels:</strong> {{ $cluster_info.Labels }}</p>
				<p><strong>Catchy Phrase:</strong> {{ $cluster_info.CatchyPhrase }}</p>
				<div class="image-container">
					{{range $index, $image := $cluster_info.Images}}
						<div class="image">
							<img src="/image/{{ $image }}" alt="Image {{ $image }}">
							<p>Product Reference ID: {{ index $cluster_info.ProductReferenceIDs $index }}</p>
						</div>
					{{end}}
				</div>
				<!-- Publish Button with Embedded Data -->
				<button onclick="publishCluster('{{ $cluster_id }}', '{{ escapeJS $cluster_info.Title }}', '{{ escapeJS $cluster_info.CatchyPhrase }}', [{{ range $i, $id := $cluster_info.ProductReferenceIDs }} '{{ $id }}'{{ if ne (add $i 1) (len $cluster_info.ProductReferenceIDs) }}, {{ end }}{{ end }}])" class="submit-button">
					Publish Cluster
				</button>
			</div>
		{{end}}
	</div>
</body>
</html>
`

	// Define template functions
	funcMap := template.FuncMap{
		"escapeJS": escapeJS,
		"add":      add,
	}

	// Parse the template with the custom functions
	t, err := template.New("clusters").Funcs(funcMap).Parse(tmpl)
	if err != nil {
		return "", fmt.Errorf("failed to parse HTML template: %v", err)
	}

	// Prepare data for the template
	data := struct {
		Clusters map[string]models.ClusterDetails
	}{
		Clusters: clusters,
	}

	// Log the data being sent to the template
	log.Println("Preparing data for HTML template:")
	for clusterID, cluster := range data.Clusters {
		log.Printf("Cluster ID: %s", clusterID)
		log.Printf("  Title: %s", cluster.Title)
		log.Printf("  Catchy Phrase: %s", cluster.CatchyPhrase)
		log.Printf("  Labels: %s", cluster.Labels)
		log.Printf("  Images: %v", cluster.Images)
		log.Printf("  Product Reference IDs: %v", cluster.ProductReferenceIDs)
	}

	// Execute the template into a buffer
	var buf bytes.Buffer
	err = t.Execute(&buf, data)
	if err != nil {
		return "", fmt.Errorf("failed to execute HTML template: %v", err)
	}

	// Define the output HTML file path
	outputFile := filepath.Join(tempDir, "clustered_fashion_items.html")

	// Write the buffer to the HTML file
	err = os.WriteFile(outputFile, buf.Bytes(), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to write HTML file: %v", err)
	}

	// Log the location of the generated HTML file
	log.Printf("HTML file generated successfully at: %s", outputFile)

	// Return the path to the HTML file
	return outputFile, nil
}

// Helper functions

// escapeJS escapes single quotes and backslashes for safe inclusion in JavaScript strings
func escapeJS(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "'", "\\'")
	return s
}

// add is a helper function to add two integers
func add(a, b int) int {
	return a + b
}

// sortClustersByID sorts the clusters based on their integer IDs
func sortClustersByID(clusters map[string]models.ClusterDetails) map[string]models.ClusterDetails {
	sorted := make(map[string]models.ClusterDetails)

	// Collect cluster IDs
	clusterIDs := make([]int, 0, len(clusters))
	idMap := make(map[int]string)
	for key := range clusters {
		var id int
		_, err := fmt.Sscanf(key, "Cluster-%d", &id)
		if err != nil {
			log.Printf("Failed to parse cluster ID '%s': %v", key, err)
			continue // Skip clusters with invalid IDs
		}
		clusterIDs = append(clusterIDs, id)
		idMap[id] = key
	}

	// Sort cluster IDs using sort.Ints for efficiency
	sort.Ints(clusterIDs)

	// Populate the sorted map
	for _, id := range clusterIDs {
		key := idMap[id]
		sorted[key] = clusters[key]
	}

	return sorted
}

// SanitizeFilename replaces invalid characters in filenames
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

// URLEncode encodes a string for safe inclusion in URLs
func URLEncode(s string) string {
	return strings.ReplaceAll(s, " ", "%20")
}

// CleanText performs basic text cleaning, such as removing extra spaces and trimming.
func CleanText(text string) string {
	return strings.TrimSpace(text)
}
