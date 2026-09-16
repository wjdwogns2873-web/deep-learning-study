#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <regex>

struct Detection {
    cv::Rect box;
    int class_id;
    float confidence;
};

class Person {
public:
    Person(std::string name) {
        std::cout << "Name: " << name << std::endl;
    }
};

class YoloInferencer {
public:
    YoloInferencer(const std::string& modelPath, const char* logid, const char* provider);
    ~YoloInferencer();
    std::vector<Detection> infer(cv::Mat& frame, float conf_threshold, float iou_threshold);

private:
    std::vector<Ort::Value> preprocess(cv::Mat& frame);
    std::vector<Ort::Value> forward(std::vector<Ort::Value>& inputTensors);
    std::vector<Detection> postprocess(std::vector<Ort::Value>& outputTensors, float conf_threshold, float iou_threshold);

    // Person person_{"DefaultName"};
    Person person_ = Person("DefaultName");

    Ort::Env env_;
    Ort::Session session_{ nullptr };

    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> inputNamesCStr_;
    std::vector<const char*> outputNamesCStr_;

    Ort::ModelMetadata model_metadata{ nullptr };
    std::unordered_map<std::string, std::string> metadata;

    std::vector<int> imgsz_;
    int stride_ = -1;
    int nc_ = -1;
    int ch_ = 3;

    std::unordered_map<int, std::string> names_;
    std::vector<int64_t> inputTensorShape_;
    std::string task_;

    std::vector<float> inputTensorValues_;

    cv::Size cvSize_;
    cv::Size rawImgSize_;

    std::vector<std::string> parseVectorString(const std::string& input);

    std::vector<int> convertStringVectorToInts(const std::vector<std::string>& input);

    std::unordered_map<int, std::string> parseNames(const std::string& input);

    int64_t vector_product(const std::vector<int64_t>& vec);
    
    static constexpr int DEFAULT_LETTERBOX_PAD_VALUE = 114;

    cv::Mat letterbox(const cv::Mat& image, const cv::Size& newShape, cv::Scalar_<double> color, 
                      bool auto_, bool scaleFill, bool scaleUp, int stride);

    std::vector<float> fill_blob(cv::Mat& image, std::vector<int64_t>& inputTensorShape);

    void clip_boxes(cv::Rect& box, const cv::Size& shape);

    void clip_boxes(cv::Rect_<float>& box, const cv::Size& shape);

    void clip_boxes(std::vector<cv::Rect>& boxes, const cv::Size& shape);

    void clip_boxes(std::vector<cv::Rect_<float>>& boxes, const cv::Size& shape);

    cv::Rect_<float> scale_boxes(const cv::Size& img1_shape, 
                                 cv::Rect_<float>& box, 
                                 const cv::Size& img0_shape,
                                 std::pair<float, cv::Point2f> ratio_pad = std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f)), 
                                 bool padding = true);
};