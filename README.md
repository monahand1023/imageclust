# ImageClust: Advanced Image Clustering and Analysis Platform

## Introduction

ImageClust is an image clustering and analysis platform that combines modern computer vision techniques with advanced machine learning algorithms to organize image collections into meaningful groups. At its core, the system leverages deep learning embeddings from ResNet50, enriched with semantic labels from AWS Rekognition, to create comprehensive image representations that capture both visual and contextual information.

The platform is designed to solve the challenging problem of organizing large image collections in a way that goes beyond simple visual similarity. By incorporating semantic understanding through AWS Rekognition and using hierarchical clustering with size constraints, ImageClust creates balanced, meaningful groups of images that are both visually and contextually related.

## Technical Architecture

### Frontend Architecture

The frontend is built as a single-page application using React 18, with a focus on performance and user experience. Here's a breakdown of the frontend components:

1. **Core Components**
   - `ImageUploadForm.jsx`: Handles file uploads with drag-and-drop functionality using browser's File API
   - `App.jsx`: Main application component managing routing and global state
   - Styling implemented using Tailwind CSS for responsive design

2. **State Management**
   - Local component state using React hooks for upload status and form data
   - File handling with proper cleanup and error management
   - Real-time progress tracking for uploads and processing



3. **API Integration**
   - RESTful API communication with the backend
   - Proper error handling and retry mechanisms
   - Progress tracking for long-running operations

### Backend Architecture

The backend is implemented in Go. The system is organized into several key packages:

1. **Core Packages**
   - `embeddings`: Handles image processing and feature extraction using ResNet50
   - `clustering`: Implements hierarchical clustering with Ward's method
   - `ai`: Manages integration with various AI services (AWS Bedrock, Rekognition)

2. **AI Service Integration**
   - AWS Rekognition for image labeling
   - AWS Bedrock integration with three AI models:
     - Claude 3.5 Haiku: Fast, efficient title generation
     - Claude 3.5 Sonnet: More sophisticated descriptions
     - Amazon Nova Micro: Alternative title generation

3. **Image Processing Pipeline**
   - OpenCV integration for image preprocessing


   - ResNet50 implementation using ONNX format
   - Efficient caching system for embeddings and labels

### Data Flow Architecture

The system processes images through several stages:



1. **Upload Stage**














   ```
   User Upload → Frontend Validation → Backend Storage
   ```

2. **Processing Stage**
   ```
   Image → Preprocessing → ResNet50 Embedding → AWS Rekognition → Combined Feature Vector
   ```

3. **Clustering Stage**
   ```
   Feature Vectors → Hierarchical Clustering → Size-Constrained Clusters
   ```

4. **Description Stage**
   ```
   Cluster Features → AWS Bedrock → Generated Titles and Descriptions
   ```

## Implementation Details

### Image Processing

The system uses a sophisticated image processing pipeline:

```go
// PreprocessImage handles image normalization and resizing
func PreprocessImage(imagePath string) (gocv.Mat, error) {
    // Load and validate image
    img := gocv.IMRead(imagePath, gocv.IMReadColor)
    
    // Resize to ResNet50 input size (224x224)
    resized := gocv.NewMat()
    gocv.Resize(img, &resized, image.Pt(224, 224), 0, 0, gocv.InterpolationLinear)
    
    // Convert to RGB and normalize
    rgb := gocv.NewMat()
    gocv.CvtColor(resized, &rgb, gocv.ColorBGRToRGB)
    
    // Create normalized blob
    return gocv.BlobFromImage(rgb, 1.0/255.0, image.Pt(224, 224), 
        gocv.NewScalar(0, 0, 0, 0), false, false)
}
```

### Clustering Algorithm

The clustering implementation uses Ward's method with size constraints:

```go
// PerformClusteringWithConstraints implements hierarchical clustering
func PerformClusteringWithConstraints(embeddings [][]float32, 
    productReferenceIDs []string, minSize, maxSize int) (map[int][]string, bool) {
    
    // Calculate optimal number of clusters
    nClusters, err := CalculateOptimalClusters(totalItems, minSize, maxSize)
    
    // Initialize clusters
    clusters := make([]Cluster, totalItems)
    for i := 0; i < totalItems; i++ {
        clusters[i] = NewCluster(i, embeddings[i])
    }
    
    // Perform hierarchical clustering
    distanceMatrix := ComputeInitialDistanceMatrix(clusters)
    for len(clusters) > nClusters {
        i, j := FindClosestClusters(distanceMatrix)
        if clusters[i].Size + clusters[j].Size <= maxSize {
            newCluster := MergeClusters(clusters[i], clusters[j])
            // Update clusters and distance matrix
        }
    }
    
    return ConvertToClusterMap(clusters, productReferenceIDs), true
}
```

## Building and Deployment

### Local Development Setup

1. **Prerequisites Installation**
   ```bash
   # Install Go dependencies
   go mod download
   
   # Install OpenCV
   wget -q https://github.com/opencv/opencv/archive/4.6.0.zip
   unzip 4.6.0.zip && cd opencv-4.6.0
   mkdir build && cd build
   cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
   make -j8
   sudo make install
   
   # Install Node.js dependencies
   cd frontend
   npm install
   ```

2. **Environment Configuration**
   Create a `.env` file:
   ```
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=us-west-2
   MODEL_PATH=/path/to/resnet50-v1-7.onnx
   ```

3. **Development Server**
   ```bash
   # Start backend
   go run main.go
   
   # Start frontend
   cd frontend
   npm start
   ```

### Docker Deployment

The project includes a multi-stage Dockerfile for optimal production deployment:

```dockerfile
# Build backend
FROM golang:1.23rc1 AS backend-builder
WORKDIR /app
COPY . .
RUN go mod download
RUN CGO_ENABLED=1 GOOS=linux go build -o imageclust main.go

# Build frontend
FROM node:18 AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Final stage
FROM debian:bookworm-slim
WORKDIR /app
COPY --from=backend-builder /app/imageclust ./
COPY --from=backend-builder /app/resnet50-v1-7.onnx ./
COPY --from=frontend-builder /app/frontend/build ./frontend/build
```

Deploy using:
```bash
docker build -t imageclust .
docker run -p 8080:8080 -v /path/to/data:/app/data imageclust
```

## Current Limitations and Future Enhancements

### Limitations

1. **Performance Constraints**
   - CPU-bound processing for ResNet50
   - Memory requirements scale with image collection size
   - Network latency for AWS service calls

2. **Clustering Constraints**
   - Fixed minimum and maximum cluster sizes
   - Cannot modify clusters after formation
   - Limited to static feature extraction

### Future Enhancements

1. **Technical Improvements**
   - Implement GPU acceleration for ResNet50
   - Add distributed processing capability
   - Introduce progressive loading for large datasets
   - Implement real-time clustering updates

2. **Feature Enhancements**
   - Custom embedding model support
   - User-defined clustering criteria
   - Interactive cluster refinement
   - Advanced visualization options
   - Export capabilities in various formats

3. **Integration Possibilities**
   - Content management system plugins
   - Cloud storage provider integration
   - API authentication and rate limiting
   - Webhook support for processing events

## Use Cases

1. **Digital Asset Management**
   - Organize product photography
   - Manage marketing assets
   - Archive historical images

2. **Content Creation**
   - Group similar content for social media
   - Organize stock photo collections
   - Manage design assets











3. **Data Analysis**
   - Visual data exploration
   - Pattern recognition in image sets
   - Content auditing

## Contributing

I welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

Please refer to our contribution guidelines for more details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project builds upon several open-source technologies and services:

- OpenCV for image processing
- React and Tailwind CSS for frontend development
- AWS SDK for cloud services integration
- Go community for excellent packages and tools

For questions, issues, or support, please open an issue in the GitHub repository.











