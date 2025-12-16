package models

import (
	"testing"
)

func TestClusterDetails_Init(t *testing.T) {
	var cd ClusterDetails
	initialized := cd.Init()

	if initialized.Images == nil {
		t.Error("expected Images to be initialized, got nil")
	}
	if len(initialized.Images) != 0 {
		t.Errorf("expected Images to be empty, got %d items", len(initialized.Images))
	}

	if initialized.ServiceOutputs == nil {
		t.Error("expected ServiceOutputs to be initialized, got nil")
	}
	if len(initialized.ServiceOutputs) != 0 {
		t.Errorf("expected ServiceOutputs to be empty, got %d items", len(initialized.ServiceOutputs))
	}
}

func TestClusterDetails_SetServiceOutput_New(t *testing.T) {
	var cd ClusterDetails
	cd = cd.Init()

	output := ServiceOutput{
		ServiceName:  "TestService",
		Title:        "Test Title",
		CatchyPhrase: "Test Phrase",
	}

	cd.SetServiceOutput(output)

	if len(cd.ServiceOutputs) != 1 {
		t.Errorf("expected 1 output, got %d", len(cd.ServiceOutputs))
	}

	if cd.ServiceOutputs[0].ServiceName != "TestService" {
		t.Errorf("expected ServiceName 'TestService', got '%s'", cd.ServiceOutputs[0].ServiceName)
	}
}

func TestClusterDetails_SetServiceOutput_Update(t *testing.T) {
	var cd ClusterDetails
	cd = cd.Init()

	// Add initial output
	cd.SetServiceOutput(ServiceOutput{
		ServiceName:  "TestService",
		Title:        "Original Title",
		CatchyPhrase: "Original Phrase",
	})

	// Update the same service
	cd.SetServiceOutput(ServiceOutput{
		ServiceName:  "TestService",
		Title:        "Updated Title",
		CatchyPhrase: "Updated Phrase",
	})

	if len(cd.ServiceOutputs) != 1 {
		t.Errorf("expected 1 output after update, got %d", len(cd.ServiceOutputs))
	}

	if cd.ServiceOutputs[0].Title != "Updated Title" {
		t.Errorf("expected Title 'Updated Title', got '%s'", cd.ServiceOutputs[0].Title)
	}
}

func TestClusterDetails_SetServiceOutput_Multiple(t *testing.T) {
	var cd ClusterDetails
	cd = cd.Init()

	cd.SetServiceOutput(ServiceOutput{ServiceName: "Service1", Title: "Title1"})
	cd.SetServiceOutput(ServiceOutput{ServiceName: "Service2", Title: "Title2"})
	cd.SetServiceOutput(ServiceOutput{ServiceName: "Service3", Title: "Title3"})

	if len(cd.ServiceOutputs) != 3 {
		t.Errorf("expected 3 outputs, got %d", len(cd.ServiceOutputs))
	}
}

func TestClusterDetails_GetOutputByServiceName_Found(t *testing.T) {
	var cd ClusterDetails
	cd = cd.Init()
	cd.SetServiceOutput(ServiceOutput{
		ServiceName:  "TargetService",
		Title:        "Found Title",
		CatchyPhrase: "Found Phrase",
	})
	cd.SetServiceOutput(ServiceOutput{
		ServiceName:  "OtherService",
		Title:        "Other Title",
		CatchyPhrase: "Other Phrase",
	})

	output, found := cd.GetOutputByServiceName("TargetService")

	if !found {
		t.Error("expected to find TargetService")
	}
	if output.Title != "Found Title" {
		t.Errorf("expected Title 'Found Title', got '%s'", output.Title)
	}
}

func TestClusterDetails_GetOutputByServiceName_NotFound(t *testing.T) {
	var cd ClusterDetails
	cd = cd.Init()
	cd.SetServiceOutput(ServiceOutput{ServiceName: "ExistingService"})

	output, found := cd.GetOutputByServiceName("NonExistentService")

	if found {
		t.Error("expected not to find NonExistentService")
	}
	if output.ServiceName != "" {
		t.Errorf("expected empty ServiceOutput, got ServiceName '%s'", output.ServiceName)
	}
}

func TestClusterDetails_GetOutputByServiceName_Empty(t *testing.T) {
	var cd ClusterDetails
	cd = cd.Init()

	_, found := cd.GetOutputByServiceName("AnyService")

	if found {
		t.Error("expected not to find any service in empty ClusterDetails")
	}
}

func TestServiceOutput_Struct(t *testing.T) {
	output := ServiceOutput{
		ServiceName:  "Claude",
		Title:        "Amazing Collection",
		CatchyPhrase: "Style meets comfort",
	}

	if output.ServiceName != "Claude" {
		t.Errorf("expected ServiceName 'Claude', got '%s'", output.ServiceName)
	}
	if output.Title != "Amazing Collection" {
		t.Errorf("expected Title 'Amazing Collection', got '%s'", output.Title)
	}
	if output.CatchyPhrase != "Style meets comfort" {
		t.Errorf("expected CatchyPhrase 'Style meets comfort', got '%s'", output.CatchyPhrase)
	}
}

func TestUploadedImage_Struct(t *testing.T) {
	data := []byte{0x89, 0x50, 0x4E, 0x47} // PNG header
	img := UploadedImage{
		Filename: "test.png",
		Data:     data,
	}

	if img.Filename != "test.png" {
		t.Errorf("expected Filename 'test.png', got '%s'", img.Filename)
	}
	if len(img.Data) != 4 {
		t.Errorf("expected Data length 4, got %d", len(img.Data))
	}
}

func TestClusterDetails_Fields(t *testing.T) {
	cd := ClusterDetails{
		Title:        "Fashion Forward",
		CatchyPhrase: "Trendy and timeless",
		Labels:       "clothing, fashion, style",
		Images:       []string{"img1.jpg", "img2.jpg"},
	}

	if cd.Title != "Fashion Forward" {
		t.Errorf("expected Title 'Fashion Forward', got '%s'", cd.Title)
	}
	if cd.CatchyPhrase != "Trendy and timeless" {
		t.Errorf("expected CatchyPhrase 'Trendy and timeless', got '%s'", cd.CatchyPhrase)
	}
	if cd.Labels != "clothing, fashion, style" {
		t.Errorf("expected Labels 'clothing, fashion, style', got '%s'", cd.Labels)
	}
	if len(cd.Images) != 2 {
		t.Errorf("expected 2 images, got %d", len(cd.Images))
	}
}
