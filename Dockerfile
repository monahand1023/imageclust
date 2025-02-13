# Build stage for React frontend
FROM node:18 AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Build stage for Go backend
FROM golang:1.23rc1 AS backend-builder
WORKDIR /app

# Install OpenCV with contrib modules
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Download and build OpenCV with contrib modules
RUN wget -q https://codeload.github.com/opencv/opencv/zip/refs/tags/4.6.0 -O opencv.zip && \
    wget -q https://codeload.github.com/opencv/opencv_contrib/zip/refs/tags/4.6.0 -O opencv_contrib.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    rm *.zip && \
    cd opencv-4.6.0 && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D OPENCV_ENABLE_NONFREE=ON \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.6.0/modules \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_opencv_python3=OFF \
          -D BUILD_opencv_python2=OFF \
          .. && \
    make -j2 && \
    make install && \
    cd ../.. && \
    rm -rf opencv-4.6.0 opencv_contrib-4.6.0

ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV CGO_CFLAGS="-I/usr/local/include/opencv4"
ENV CGO_LDFLAGS="-L/usr/local/lib -lopencv_core -lopencv_face -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_features2d -lopencv_video -lopencv_dnn -lopencv_xfeatures2d -lopencv_plot -lopencv_tracking -lopencv_img_hash -lopencv_calib3d -lopencv_bgsegm -lopencv_aruco"

COPY go.* ./
COPY . .
RUN go mod download
RUN CGO_ENABLED=1 GOOS=linux go build -o imageclust main.go 2>&1 | tee build.log || (cat build.log && exit 1)

# Final runtime stage
FROM debian:bookworm-slim
WORKDIR /app

COPY --from=backend-builder /usr/local/lib/libopencv_* /usr/local/lib/
COPY --from=backend-builder /usr/local/lib/pkgconfig/opencv4.pc /usr/local/lib/pkgconfig/

RUN apt-get update && \
    apt-get install -y libgtk2.0-dev pkg-config && \
    rm -rf /var/lib/apt/lists/* && \
    ldconfig

COPY --from=backend-builder /app/imageclust ./
COPY --from=backend-builder /app/resnet50-v1-7.onnx ./
COPY --from=frontend-builder /app/frontend/build ./frontend/build

EXPOSE 8080

CMD ["./imageclust"]