package models

type UploadedImage struct {
	Filename string
	Data     []byte
}

type ClusterDetails struct {
	Title        string
	CatchyPhrase string
	Images       []string
}
