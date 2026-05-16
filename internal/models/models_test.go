package models

import "testing"

func TestUploadedImage(t *testing.T) {
	img := UploadedImage{
		Filename: "test.jpg",
		Data:     []byte{0xFF, 0xD8, 0xFF},
	}
	if img.Filename != "test.jpg" {
		t.Errorf("Filename = %q, want %q", img.Filename, "test.jpg")
	}
	if len(img.Data) != 3 {
		t.Errorf("len(Data) = %d, want 3", len(img.Data))
	}
}

func TestClusterDetails(t *testing.T) {
	cd := ClusterDetails{
		Title:        "Summer Vibes",
		CatchyPhrase: "Bright days ahead",
		Images:       []string{"a.jpg", "b.jpg"},
	}
	if cd.Title != "Summer Vibes" {
		t.Errorf("Title = %q", cd.Title)
	}
	if len(cd.Images) != 2 {
		t.Errorf("len(Images) = %d, want 2", len(cd.Images))
	}
}
