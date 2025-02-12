// Package utils
package utils

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html/template"
	"imageclust/internal/models"
	"os"
	"path/filepath"
	"strings"
)

type ClusterDownload struct {
	Title        string   `json:"title"`
	CatchyPhrase string   `json:"catchyPhrase"`
	Images       []string `json:"images"`
	Labels       string   `json:"labels"`
}

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
        .download-button {
            background-color: #4CAF50;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
            font-size: 0.9em;
        }
        .download-button:hover {
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
        async function downloadCluster(clusterId, title, catchyPhrase, images, labels) {
            const clusterData = {
                title: title,
                catchyPhrase: catchyPhrase,
                images: images,
                labels: labels
            };
            
            const blob = new Blob([JSON.stringify(clusterData, null, 2)], { type: 'application/json' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'cluster-' + clusterId + '.json';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>Model Comparison</h1>
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
                                    <button onclick="downloadCluster('{{ $cluster_id }}', '{{ escapeJS $output.Title }}', '{{ escapeJS $output.CatchyPhrase }}', {{escapeJS (toJSON $cluster_info.Images)}}, '{{ escapeJS $cluster_info.Labels }}')" class="download-button">
                                        Download Cluster
                                    </button>
                                </td>
                            </tr>
                        {{end}}
                    </tbody>
                </table>

                <div class="image-container">
                    {{range $index, $image := $cluster_info.Images}}
                        <div class="image">
                            <img src="/api/image/{{$image}}" alt="Cluster image">
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
		"toJSON":   toJSON,
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

	// Execute the template into a buffer
	var buf bytes.Buffer
	err = t.Execute(&buf, data)
	if err != nil {
		return "", fmt.Errorf("failed to execute HTML template: %v", err)
	}

	// Define the output HTML file path
	outputFile := filepath.Join(tempDir, "clusters.html")

	// Write the buffer to the HTML file
	err = os.WriteFile(outputFile, buf.Bytes(), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to write HTML file: %v", err)
	}

	return outputFile, nil
}

// Helper functions
func escapeJS(s interface{}) string {
	switch v := s.(type) {
	case string:
		v = strings.ReplaceAll(v, "\\", "\\\\")
		v = strings.ReplaceAll(v, "'", "\\'")
		return v
	default:
		return ""
	}
}

func toJSON(v interface{}) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "[]"
	}
	return string(b)
}

func add(a, b int) int {
	return a + b
}

func SanitizeFilename(name string) string {
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

func URLEncode(s string) string {
	return strings.ReplaceAll(s, " ", "%20")
}

func CleanText(text string) string {
	return strings.TrimSpace(text)
}
