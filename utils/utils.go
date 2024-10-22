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
	"strings"
)

// GenerateHTMLOutput generates an HTML file representing the clusters.
func GenerateHTMLOutput(clusterDetails map[string]models.ClusterDetails, tempDir, host string, port int) (string, error) {
	// Define the HTML template similar to your Python Jinja2 template
	const tmpl = `
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Clustered Fashion Items</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f7f7f7; }
            .container { max-width: 1200px; margin: 0 auto; }
            .controls { margin-bottom: 20px; padding: 20px; background-color: #f0f0f0; border-radius: 8px; }
            .control-table { width: 100%; border-collapse: collapse; }
            .control-table th, .control-table td { padding: 10px; border-bottom: 1px solid #ddd; }
            .control-table th { text-align: left; color: #34495e; font-size: 18px; }
            .cluster { background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); margin-bottom: 40px; padding: 20px; }
            .image-container { display: flex; flex-wrap: wrap; }
            .image { margin: 10px; flex: 1 1 150px; }
            img { max-width: 150px; height: auto; border-radius: 4px; border: 1px solid #ddd; }
            h1 { color: #34495e; text-align: center; margin-bottom: 40px; }
            h2 { color: #34495e; font-size: 20px; margin-bottom: 5px; }
            h3 { color: #2c3e50; font-size: 16px; }
            .catchy-phrase, .labels { margin-bottom: 10px; }
            .labels { font-size: 14px; color: #7f8c8d; }
            .submit-button { padding: 10px 20px; background-color: #2980b9; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
            .submit-button:hover { background-color: #1c5980; }
            .message { padding: 10px; margin-bottom: 20px; border-radius: 4px; }
            .success { background-color: #d4edda; color: #155724; }
            .error { background-color: #f8d7da; color: #721c24; }
        </style>
        <script>
            async function submitWeights(event) {
                event.preventDefault(); // Prevent the form from submitting the traditional way

                let formData = new FormData(event.target);
                fetch("http://{{ .ServerAddress }}:{{ .ServerPort }}/update_weights", {
                    method: "POST",
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        window.location.href = data.redirect_url;
                    } else {
                        alert("Error updating weights: " + data.error);
                    }
                })
                .catch(error => console.error("Error:", error));
            }

            async function publishCluster(clusterId) {
                const clusterData = { cluster_id: clusterId };
            
                try {
                    const response = await fetch("http://{{ .ServerAddress }}:{{ .ServerPort }}/publish", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify(clusterData)
                    });
                    const data = await response.json();
                    if (data.success) {
                        alert("Cluster published successfully!");
                    } else {
                        alert("Failed to publish cluster: " + data.error);
                    }
                } catch (error) {
                    console.error("Error:", error);
                    alert("An error occurred.");
                }
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Clustered Fashion Items</h1>
            <!-- Clusters -->
            {{range $cluster_id, $cluster_info := .SortedClusters}}
                <div class="cluster">
                    <h2>{{ $cluster_info.Title }}</h2>
                    <h3>Cluster ID: {{ $cluster_id }}</h3>
                    <p class="catchy-phrase"><strong>Catchy Phrase:</strong> {{ $cluster_info.CatchyPhrase }}</p>
                    <p class="labels"><strong>Labels:</strong> {{ $cluster_info.Labels }}</p>
                    <div class="image-container">
                        {{range $index, $image := $cluster_info.Images}}
                            <div class="image">
                                <img src="file://{{ $.TempDir }}/images/{{ $image }}" alt="Image {{ $image }}">
                                <p>Product Reference ID: {{ index $cluster_info.ProductReferenceIDs $index }}</p>
                            </div>
                        {{end}}
                    </div>
                    <!-- Publish Button -->
                    <button onclick="publishCluster('{{ $cluster_id }}')" class="submit-button">
                        Publish Cluster
                    </button>
                </div>
            {{end}}
        </div>
    </body>
    </html>
    `

	// Parse the template
	t, err := template.New("clusters").Parse(tmpl)
	if err != nil {
		return "", fmt.Errorf("failed to parse HTML template: %v", err)
	}

	// Prepare data for the template
	data := struct {
		SortedClusters map[string]models.ClusterDetails
		ServerAddress  string
		ServerPort     int
		TempDir        string // Added TempDir
	}{
		SortedClusters: sortClustersByID(clusterDetails),
		ServerAddress:  host,
		ServerPort:     port,
		TempDir:        tempDir, // Pass tempDir to the template
	}

	// Log the data being sent to the template
	log.Println("Preparing data for HTML template:")
	for clusterID, cluster := range data.SortedClusters {
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
	log.Printf("HTML file generated successfully at: file://%s\n", outputFile)

	return outputFile, nil
}

// sortClustersByID sorts the clusters based on their integer IDs
func sortClustersByID(clusters map[string]models.ClusterDetails) map[string]models.ClusterDetails {
	sorted := make(map[string]models.ClusterDetails)

	// Collect and sort the cluster IDs numerically
	clusterIDs := make([]int, 0, len(clusters))
	idMap := make(map[int]string)
	for key := range clusters {
		var id int
		fmt.Sscanf(key, "Cluster-%d", &id)
		clusterIDs = append(clusterIDs, id)
		idMap[id] = key
	}

	// Sort the cluster IDs
	for i := 0; i < len(clusterIDs)-1; i++ {
		for j := i + 1; j < len(clusterIDs); j++ {
			if clusterIDs[i] > clusterIDs[j] {
				clusterIDs[i], clusterIDs[j] = clusterIDs[j], clusterIDs[i]
			}
		}
	}

	// Populate the sorted map
	for _, id := range clusterIDs {
		key := idMap[id]
		sorted[key] = clusters[key]
	}

	return sorted
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

// URLEncode encodes a string for safe inclusion in URLs
func URLEncode(s string) string {
	return strings.ReplaceAll(s, " ", "%20")
}

// CleanText performs basic text cleaning, such as removing extra spaces and trimming.
func CleanText(text string) string {
	return strings.TrimSpace(text)
}
