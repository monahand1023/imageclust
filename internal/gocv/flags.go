package gocv

// #cgo linux pkg-config: opencv4
// #cgo CXXFLAGS: -std=c++11
// #cgo LDFLAGS: -L/usr/local/lib -lopencv_core -lopencv_face -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_features2d -lopencv_video -lopencv_dnn -lopencv_xfeatures2d -lopencv_plot -lopencv_tracking -lopencv_img_hash -lopencv_calib3d -lopencv_bgsegm -lopencv_aruco
import "C"
