package productsetter

import (
	"fmt"
	"net/http"
	"strconv"
)

// AppConfig holds the configuration extracted from the request.
type AppConfig struct {
	ProfileID         string
	AuthToken         string
	NumberOfDaysLimit int
	ModelPath         string
	Host              string
	Port              int
	MinClusterSize    int
	MaxClusterSize    int
}

// ExtractConfigurations parses the configuration data from the request.
func ExtractConfigurations(r *http.Request) (*AppConfig, error) {
	appCtx := &AppConfig{}

	// Extract ProfileID
	profileID := r.FormValue("profile_id")
	if profileID == "" {
		return nil, fmt.Errorf("missing 'profile_id' field")
	}
	appCtx.ProfileID = profileID

	// Extract AuthToken
	authToken := r.FormValue("auth_token")
	if authToken == "" {
		return nil, fmt.Errorf("missing 'auth_token' field")
	}
	appCtx.AuthToken = authToken

	// Extract NumberOfDaysLimit
	numberOfDaysLimitStr := r.FormValue("number_of_days_limit")
	numberOfDaysLimit, err := strconv.Atoi(numberOfDaysLimitStr)
	if err != nil || numberOfDaysLimit <= 0 {
		appCtx.NumberOfDaysLimit = 30 // Default value
	} else {
		appCtx.NumberOfDaysLimit = numberOfDaysLimit
	}

	// Extract Host
	host := r.FormValue("host")
	if host == "" {
		appCtx.Host = "localhost" // Default value
	} else {
		appCtx.Host = host
	}

	// Extract Port
	portStr := r.FormValue("port")
	port, err := strconv.Atoi(portStr)
	if err != nil || port <= 0 {
		appCtx.Port = 5003 // Default value
	} else {
		appCtx.Port = port
	}

	// Extract MinClusterSize
	minClusterSizeStr := r.FormValue("min_cluster_size")
	minClusterSize, err := strconv.Atoi(minClusterSizeStr)
	if err != nil || minClusterSize <= 0 {
		appCtx.MinClusterSize = 3 // Default value
	} else {
		appCtx.MinClusterSize = minClusterSize
	}

	// Extract MaxClusterSize
	maxClusterSizeStr := r.FormValue("max_cluster_size")
	maxClusterSize, err := strconv.Atoi(maxClusterSizeStr)
	if err != nil || maxClusterSize <= 0 {
		appCtx.MaxClusterSize = 6 // Default value
	} else {
		appCtx.MaxClusterSize = maxClusterSize
	}

	return appCtx, nil
}
