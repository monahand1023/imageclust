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

	appCtx.MaxClusterSize = 6
	appCtx.MinClusterSize = 3

	return appCtx, nil
}
