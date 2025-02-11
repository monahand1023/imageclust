// Package utils
package utils

import (
	"bytes"
	"fmt"
	"html/template"
	"imageclust/models"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// GenerateHTMLOutput generates an HTML file based on cluster details.
func GenerateHTMLOutput(clusters map[string]models.ClusterDetails, tempDir string) (string, error) {
	const tmpl = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model Comparison - Clustered Fashion Items</title>
    <style>
        .container {
            width: 95%;
            margin: auto;
            padding: 20px;
        }
        .cluster {
            border: 1px solid #ccc;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            background: #fff;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }
        .comparison-table th {
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
            color: #2c3e50;
        }
        .comparison-table td {
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
            vertical-align: top;
        }
        .comparison-table tr:hover {
            background-color: #f8f9fa;
        }
        .image-container {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 20px;
        }
        .image {
            text-align: center;
            flex: 0 0 200px;
        }
        .image img {
            max-width: 200px;
            height: auto;
            border-radius: 4px;
        }
        .submit-button {
            background-color: #4CAF50;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
            font-size: 0.9em;
        }
        .submit-button:hover {
            background-color: #45a049;
        }
        .labels {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 15px;
            font-size: 0.9em;
        }
        .product-id {
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }
        .model-name {
            font-weight: 500;
            color: #2c3e50;
        }
    </style>
    <script>
        async function publishCluster(clusterId, title, catchyPhrase, productReferenceIds, modelName) {
            const payload = {
                cluster_id: clusterId,
                title: title,
                description: catchyPhrase,
                product_reference_ids: productReferenceIds,
                model_name: modelName
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
                    alert("Cluster published successfully using " + modelName + "!");
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
        <h1>Model Comparison - Clustered Fashion Items</h1>
        {{range $cluster_id, $cluster_info := .Clusters}}
            <div class="cluster">
                <div class="labels">
                    <strong>Labels:</strong> {{ $cluster_info.Labels }}
                </div>
                
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Title</th>
                            <th>Catchy Phrase</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {{range $output := $cluster_info.ServiceOutputs}}
                            <tr>
                                <td class="model-name">{{ $output.ServiceName }}</td>
                                <td>{{ $output.Title }}</td>
                                <td>{{ $output.CatchyPhrase }}</td>
                                <td>
                                    <button onclick="publishCluster('{{ $cluster_id }}', '{{ escapeJS $output.Title }}', '{{ escapeJS $output.CatchyPhrase }}', [{{range $i, $id := $cluster_info.ProductReferenceIDs}}'{{ $id }}'{{if ne (add $i 1) (len $cluster_info.ProductReferenceIDs)}}, {{end}}{{end}}], '{{ $output.ServiceName }}')" class="submit-button">
                                        Publish
                                    </button>
                                </td>
                            </tr>
                        {{end}}
                    </tbody>
                </table>

                <div class="image-container">
                    {{range $index, $image := $cluster_info.Images}}
                        <div class="image">
                            <img src="/image/{{ $image }}" alt="Product Image">
                            <p class="product-id">ID: {{ index $cluster_info.ProductReferenceIDs $index }}</p>
                        </div>
                    {{end}}
                </div>
            </div>
        {{end}}
    </div>
</body>
</html>`

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
