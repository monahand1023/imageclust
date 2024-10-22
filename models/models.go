package models

import (
	"log"
	"time"
)

// CombinedProductDetails represents the combined details of a product.
type CombinedProductDetails struct {
	ProductReferenceID string   `json:"product_reference_id"`
	AdvertiserID       string   `json:"advertiser_id"`
	Price              float32  `json:"price"`
	ImagePath          string   `json:"image_path"`
	Labels             []string `json:"labels"`
	Description        string   `json:"description,omitempty"`
	Title              string   `json:"title,omitempty"`
	UpdatedAt          string   `json:"updated_at,omitempty"`
}

// ClusterDetails represents the details of a single cluster.
type ClusterDetails struct {
	Title               string
	CatchyPhrase        string
	Labels              string
	Images              []string
	ProductReferenceIDs []string
}

// NewCombinedProductDetails creates a new CombinedProductDetails instance.
func NewCombinedProductDetails(productReferenceID, advertiserID string, price float32, imagePath, description, title, updatedAt string) *CombinedProductDetails {
	return &CombinedProductDetails{
		ProductReferenceID: productReferenceID,
		AdvertiserID:       advertiserID,
		Price:              price,
		ImagePath:          imagePath,
		Labels:             []string{},
		Description:        description,
		Title:              title,
		UpdatedAt:          updatedAt,
	}
}

// PrepareLabelsMapping generates a map for product labels.

// PreparePriceMapping generates a map for product prices.

// PrepareAdvertiserMapping generates a map for advertiser IDs.

// PrepareTitleMapping generates a map for product titles.

// PrepareDescriptionMapping generates a map for product descriptions.

// PrepareUpdatedAtMapping generates a map for updated_at timestamps.

// GetFormattedUpdatedAt formats the updated_at field to YYYY-MM-DD.
func (p *CombinedProductDetails) GetFormattedUpdatedAt() string {
	if p.UpdatedAt == "" {
		return ""
	}
	parsedTime, err := time.Parse(time.RFC3339, p.UpdatedAt)
	if err != nil {
		log.Printf("Error formatting updated_at for product %s: %v", p.ProductReferenceID, err)
		return ""
	}
	return parsedTime.Format("2006-01-02")
}

// ProductDetailsMap retrieves a product's details by its reference ID.
func ProductDetailsMap(pid string, productDetails []CombinedProductDetails) *CombinedProductDetails {
	for _, product := range productDetails {
		if product.ProductReferenceID == pid {
			return &product
		}
	}
	return nil
}
