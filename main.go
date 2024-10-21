// main.go
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
	//	router.HandleFunc("/publish", handlers.PublishHandler).Methods("POST")
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
<!-- main.go (assuming it serves the HTML form) -->
<!DOCTYPE html>
<html>
<head>
    <title>Product Clustering</title>
</head>
<body>
    <h1>Cluster Products</h1>
    <form action="/cluster_and_generate" method="post" enctype="multipart/form-data">
        <label for="profile_id">Profile ID:</label><br>
        <input type="text" id="profile_id" name="profile_id" required value="b5815d12-50e5-11ee-8376-940c556626de"><br><br>

        <label for="auth_token">Auth Token:</label><br>
        <input type="text" id="auth_token" name="auth_token" required><br><br>

        <label for="number_of_days_limit">Number of Days Limit:</label><br>
        <input type="number" id="number_of_days_limit" name="number_of_days_limit" min="1" value="30" required><br><br>

        <input type="submit" value="Cluster and Generate">
    </form>
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
