package ai

import (
	"testing"
)

func TestClaudeHaikuServiceConstant(t *testing.T) {
	if ClaudeHaikuService != 1 {
		t.Errorf("ClaudeHaikuService = %d, want 1", ClaudeHaikuService)
	}
}

func TestServiceConfigStruct(t *testing.T) {
	config := ServiceConfig{
		ServiceType: ClaudeHaikuService,
		Name:        "Test Service",
		Model:       nil,
		Order:       1,
	}

	if config.ServiceType != ClaudeHaikuService {
		t.Errorf("ServiceType = %d, want %d", config.ServiceType, ClaudeHaikuService)
	}
	if config.Name != "Test Service" {
		t.Errorf("Name = %s, want 'Test Service'", config.Name)
	}
	if config.Order != 1 {
		t.Errorf("Order = %d, want 1", config.Order)
	}
}

func TestModelOutputStruct(t *testing.T) {
	output := ModelOutput{
		ServiceName:  "Claude Haiku",
		Title:        "Test Title",
		CatchyPhrase: "Test Phrase",
		Order:        2,
	}

	if output.ServiceName != "Claude Haiku" {
		t.Errorf("ServiceName = %s, want 'Claude Haiku'", output.ServiceName)
	}
	if output.Title != "Test Title" {
		t.Errorf("Title = %s, want 'Test Title'", output.Title)
	}
	if output.CatchyPhrase != "Test Phrase" {
		t.Errorf("CatchyPhrase = %s, want 'Test Phrase'", output.CatchyPhrase)
	}
	if output.Order != 2 {
		t.Errorf("Order = %d, want 2", output.Order)
	}
}

func TestAvailableServicesConfiguration(t *testing.T) {
	if len(AvailableServices) < 1 {
		t.Error("AvailableServices should have at least one service")
	}

	// Check that Claude Haiku is configured
	found := false
	for _, svc := range AvailableServices {
		if svc.ServiceType == ClaudeHaikuService {
			found = true
			if svc.Name == "" {
				t.Error("Claude Haiku service should have a name")
			}
		}
	}
	if !found {
		t.Error("Claude Haiku service should be in AvailableServices")
	}
}

func TestGenerateTitleAndCatchyPhrase_UnknownService(t *testing.T) {
	// Test with unknown service type (should return defaults)
	title, phrase := GenerateTitleAndCatchyPhrase("test text", 1, 999)

	if title != "No Title" {
		t.Errorf("expected title 'No Title' for unknown service, got '%s'", title)
	}
	if phrase != "No Catchy Phrase" {
		t.Errorf("expected phrase 'No Catchy Phrase' for unknown service, got '%s'", phrase)
	}
}

func TestModelOutput_ZeroValue(t *testing.T) {
	var output ModelOutput

	if output.ServiceName != "" {
		t.Errorf("zero value ServiceName should be empty, got '%s'", output.ServiceName)
	}
	if output.Title != "" {
		t.Errorf("zero value Title should be empty, got '%s'", output.Title)
	}
	if output.CatchyPhrase != "" {
		t.Errorf("zero value CatchyPhrase should be empty, got '%s'", output.CatchyPhrase)
	}
	if output.Order != 0 {
		t.Errorf("zero value Order should be 0, got %d", output.Order)
	}
}

func TestServiceConfig_ZeroValue(t *testing.T) {
	var config ServiceConfig

	if config.ServiceType != 0 {
		t.Errorf("zero value ServiceType should be 0, got %d", config.ServiceType)
	}
	if config.Name != "" {
		t.Errorf("zero value Name should be empty, got '%s'", config.Name)
	}
	if config.Model != nil {
		t.Error("zero value Model should be nil")
	}
	if config.Order != 0 {
		t.Errorf("zero value Order should be 0, got %d", config.Order)
	}
}

func TestAvailableServices_OrderIsSet(t *testing.T) {
	for _, svc := range AvailableServices {
		if svc.Order < 1 {
			t.Errorf("service %s should have Order >= 1, got %d", svc.Name, svc.Order)
		}
	}
}
