package main

import (
	"log"
	"time"
)

// CombinedProductDetails represents the combined details of a product.
type CombinedProductDetails struct {
	ProductReferenceID string   `json:"product_reference_id"`
	AdvertiserID       string   `json:"advertiser_id"`
	Price              float64  `json:"price"`
	ImagePath          string   `json:"image_path"`
	Labels             []string `json:"labels"`
	Description        string   `json:"description,omitempty"`
	Title              string   `json:"title,omitempty"`
	UpdatedAt          string   `json:"updated_at,omitempty"`
}

// NewCombinedProductDetails creates a new CombinedProductDetails instance.
func NewCombinedProductDetails(productReferenceID, advertiserID string, price float64, imagePath, description, title, updatedAt string) *CombinedProductDetails {
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
func PrepareLabelsMapping(products []CombinedProductDetails, includeLabels bool) map[string][]string {
	labelsMapping := make(map[string][]string)
	for _, product := range products {
		if includeLabels && len(product.Labels) == 0 {
			log.Printf("Product %s is missing labels.", product.ProductReferenceID)
		}
		labelsMapping[product.ProductReferenceID] = product.Labels
	}
	return labelsMapping
}

// PreparePriceMapping generates a map for product prices.
func PreparePriceMapping(products []CombinedProductDetails, includePrice bool) map[string]float64 {
	priceMapping := make(map[string]float64)
	for _, product := range products {
		if includePrice && product.Price <= 0 {
			log.Printf("Product %s is missing a valid price.", product.ProductReferenceID)
		}
		priceMapping[product.ProductReferenceID] = product.Price
	}
	return priceMapping
}

// PrepareAdvertiserMapping generates a map for advertiser IDs.
func PrepareAdvertiserMapping(products []CombinedProductDetails, includeAdvertiser bool) map[string]string {
	advertiserMapping := make(map[string]string)
	for _, product := range products {
		if includeAdvertiser && product.AdvertiserID == "" {
			log.Printf("Product %s is missing an advertiser ID.", product.ProductReferenceID)
		}
		advertiserMapping[product.ProductReferenceID] = product.AdvertiserID
	}
	return advertiserMapping
}

// PrepareTitleMapping generates a map for product titles.
func PrepareTitleMapping(products []CombinedProductDetails, includeTitle bool) map[string]string {
	titleMapping := make(map[string]string)
	for _, product := range products {
		if includeTitle && product.Title == "" {
			log.Printf("Product %s is missing a title.", product.ProductReferenceID)
		}
		titleMapping[product.ProductReferenceID] = product.Title
	}
	return titleMapping
}

// PrepareDescriptionMapping generates a map for product descriptions.
func PrepareDescriptionMapping(products []CombinedProductDetails, includeDescription bool) map[string]string {
	descriptionMapping := make(map[string]string)
	for _, product := range products {
		if includeDescription && product.Description == "" {
			log.Printf("Product %s is missing a description.", product.ProductReferenceID)
		}
		descriptionMapping[product.ProductReferenceID] = product.Description
	}
	return descriptionMapping
}

// PrepareUpdatedAtMapping generates a map for updated_at timestamps.
func PrepareUpdatedAtMapping(products []CombinedProductDetails, includeUpdatedAt bool) map[string]string {
	updatedAtMapping := make(map[string]string)
	for _, product := range products {
		if includeUpdatedAt && product.UpdatedAt == "" {
			log.Printf("Product %s is missing an updated_at date.", product.ProductReferenceID)
		}
		updatedAtMapping[product.ProductReferenceID] = product.GetFormattedUpdatedAt()
	}
	return updatedAtMapping
}

// PrepareImageFilenameMapping generates a map for image filenames.
func PrepareImageFilenameMapping(products []CombinedProductDetails) map[string]string {
	imageFilenameMapping := make(map[string]string)
	for _, product := range products {
		imageFilenameMapping[product.ProductReferenceID] = product.ImagePath
	}
	return imageFilenameMapping
}

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
