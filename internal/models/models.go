package models

// ServiceOutput represents the output from a single AI service
type ServiceOutput struct {
	ServiceName  string
	Title        string
	CatchyPhrase string
}

type UploadedImage struct {
	Filename string
	Data     []byte
}

// ClusterDetails represents the details of a single cluster.
type ClusterDetails struct {
	Title          string
	CatchyPhrase   string
	Labels         string
	Images         []string
	ServiceOutputs []ServiceOutput // New field for multiple service outputs
}

func (c *ClusterDetails) Init() ClusterDetails {
	return ClusterDetails{
		Images:         make([]string, 0),
		ServiceOutputs: make([]ServiceOutput, 0),
	}
}

// GetOutputByServiceName retrieves the output for a specific service from a cluster
func (c *ClusterDetails) GetOutputByServiceName(serviceName string) (ServiceOutput, bool) {
	for _, output := range c.ServiceOutputs {
		if output.ServiceName == serviceName {
			return output, true
		}
	}
	return ServiceOutput{}, false
}

// SetServiceOutput adds or updates the output for a specific service
func (c *ClusterDetails) SetServiceOutput(output ServiceOutput) {
	// Update existing output if found
	for i, existing := range c.ServiceOutputs {
		if existing.ServiceName == output.ServiceName {
			c.ServiceOutputs[i] = output
			return
		}
	}
	// Add new output if not found
	c.ServiceOutputs = append(c.ServiceOutputs, output)
}
