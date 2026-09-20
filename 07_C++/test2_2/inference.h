#pragma once

#define RET_OK nullptr

#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

enum MODEL_TYPE
{
    // float 32 model
    YOLO_DETECT = 1, 
    YOLO_POSE = 2, 
    YOLO_CLS = 3, 

    // float 16 model
    YOLO_DETECT_HALF = 4, 
    YOLO_POSE_HALF = 5, 
    YOLO_CLS_HALF = 6
};

typedef struct _DL_INIT_PARAM
{
    std::string modelPath;
    MODEL_TYPE modelType = YOLO_DETECT;
    std::vector<int> imgSize = { 640, 640 };
    float confThreshold = 0.6f;
    float iouThreshold = 0.5f;
    int keyPointsNum = 2;
    bool cudaEnable = false;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
} DL_INIT_PARAM;

typedef struct _DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
} DL_RESULT;

class YOLO_V8
{
public:
    YOLO_V8();
    ~YOLO_V8();

    const char* CreateSession(DL_INIT_PARAM& iParams);
    const char* WarmUpSession();
    const char* RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult);

    template<typename N>
    const char* TensorProcess(clock_t starttime_1, 
                              cv::Mat& iImg, 
                              N& blob, 
                              std::vector<int64_t>& inputNodeDims, 
                              std::vector<DL_RESULT>& oResult);

    const char* PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);

    std::vector<std::string> classes{};

private:
    Ort::Env env;
    Ort::Session* session{ nullptr };
    bool cudaEnable{ false };
    Ort::RunOptions options;
    
    std::vector<std::string> inputNodeNames;
    std::vector<std::string> outputNodeNames;
    std::vector<const char*> inputNodeNamesCStr;
    std::vector<const char*> outputNodeNamesCStr;

    MODEL_TYPE modelType;
    std::vector<int> imgSize;
    float confThreshold;
    float iouThreshold;
    float resizeScales = 1.0f;
};