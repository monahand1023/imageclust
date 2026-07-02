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

// SkippedImage describes an uploaded image that could not be processed
// (e.g. an undecodable format) and was excluded from clustering.
type SkippedImage struct {
	Filename string `json:"filename"`
	Error    string `json:"error"`
}
