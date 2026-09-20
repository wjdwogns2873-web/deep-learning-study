#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <algorithm>

/**
바운딩박스를 나타내는 구조체
box에는 (x, y, w, h)가 있다. 각각 xmin, ymin, boxW, boxH
 */
struct Detection {
    cv::Rect box;
    int classId;
    float confidence;
};

class YOLOv8Detector {
public:
    YOLOv8Detector(const std::string& modelPath, const std::string& logid, const std::string& provider);
    ~YOLOv8Detector();
    std::vector<Detection> infer(cv::Mat& frame, float conf_threshold, float iou_threshold);
    cv::Mat drawDetections(const cv::Mat& frame, const std::vector<Detection>& detections);
private:
    // 전처리. frame이 들어오면 Ort::Value를 반환? 왜 벡터지.? std::vector<cv::Mat>도 아니고 한 프레임만 들어가는데?
    std::vector<Ort::Value> preprocess(cv::Mat& frame);
    std::vector<Ort::Value> forward(std::vector<Ort::Value>& inputTensors);
    std::vector<Detection> postprocess(std::vector<Ort::Value>& outputTensors, 
                                                                float conf_threshold, 
                                                                float iou_threshold);
    Ort::Env env_;
    Ort::Session session_{nullptr};

    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> inputNamesCStr_;
    std::vector<const char*> outputNamesCStr_;

    Ort::ModelMetadata modelMetadata_;
    std::unordered_map<std::string, std::string> metadataMap_;

    int stride_ = -1;
    int nc_ = -1;
    int ch_ = 3;

    // {0: 'Person', 1: 'Apple'} 이 담길 변수
    std::unordered_map<int, std::string> names_;
    std::vector<int64_t> inputTensorShape_;
    std::string task_;

    std::vector<float> inputTensorValues_;
    
    cv::Size cvSize_;
    cv::Size orgImgSize_;

    float scaleRatio_ = -1.0;
    float padX_ = -1.0;
    float padY_ = -1.0;

    // std::vector<std::string> parseVectorString(const std::string& input);
    // std::vector<int> convertStringVectorToInts(const std::vector<std::string>& input);
    std::unordered_map<int, std::string> parseNames(const std::string& input);
    int64_t vector_product(const std::vector<int64_t>& shape);

    static constexpr int DEFAULT_LETTERBOX_PAD_VALUE = 114;

    cv::Mat letterbox(const cv::Mat& rgbFrame, 
                      cv::Scalar_<double> color, 
                      bool auto_, 
                      bool scaleFill, 
                      bool scaleUp, 
                      int stride);

    std::vector<float> imageToBlob(cv::Mat& letterboxImage);

    void clip_boxes(cv::Rect_<float>& box);
    void clip_boxes_for_many_frames(std::vector<cv::Rect_<float>>& boxes);

    cv::Rect_<float> scale_boxes(cv::Rect_<float>& box, bool padding = true);
};