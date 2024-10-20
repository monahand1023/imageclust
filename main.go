package main

import (
	"log"
	"net/http"

	"ProductSetter/handlers"

	"github.com/gorilla/mux"
	"html/template"
)

func main() {
	// Initialize the router using Gorilla Mux
	router := mux.NewRouter()

	// Register handlers for various routes
	router.HandleFunc("/cluster_and_generate", handlers.ClusterAndGenerateHandler).Methods("POST")
	router.HandleFunc("/publish", handlers.PublishHandler).Methods("POST")
	router.HandleFunc("/generate_html_output", handlers.GenerateHTMLOutputHandler).Methods("POST")
	router.HandleFunc("/", htmlFormHandler).Methods("GET") // Serve the HTML form on the root path

	// Define the server address and port
	serverAddress := ":8080" // You can change the port as needed

	// Log the server start
	log.Printf("Starting server on %s", serverAddress)

	// Start the HTTP server
	err := http.ListenAndServe(serverAddress, router)
	if err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

// Handler to serve the HTML form
func htmlFormHandler(w http.ResponseWriter, r *http.Request) {
	// Template for the HTML form
	formTemplate := `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Go Server Front-End</title>
</head>
<body>
<h1 align="center">Clustering Service Demo</h1>
<form id="dataForm" enctype="multipart/form-data">
    <label for="authToken">Auth Token:</label>
    <input type="text" id="authToken" name="auth_token" required>
    <br><br>

    <label for="profileId">Profile ID:</label>
    <input type="text" id="profileId" name="profile_id" value="b5815d12-50e5-11ee-8376-940c556626de" required>
    <br><br>

    <label for="numberOfDays">Number of Days of Data:</label>
    <input type="number" id="numberOfDays" name="numberOfDays" value="200" min="1" required>
    <br><br>

    <label for="images">Upload Images:</label>
    <input type="file" id="images" name="images" multiple>
    <br><br>

    <button type="button" onclick="fetchData()">Cluster and Generate</button>
</form>

<pre id="response"></pre>

<script>
    function fetchData() {
        const form = document.getElementById('dataForm');
        const formData = new FormData(form);

        fetch('/cluster_and_generate', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('response').textContent = JSON.stringify(data, null, 2);
        })
        .catch(error => console.error('Error:', error));
    }
</script>
</body>
</html>
`

	// Parse and execute the template
	tmpl := template.New("form")
	tmpl, err := tmpl.Parse(formTemplate)
	if err != nil {
		http.Error(w, "Unable to load form", http.StatusInternalServerError)
		return
	}
	tmpl.Execute(w, nil)
}
