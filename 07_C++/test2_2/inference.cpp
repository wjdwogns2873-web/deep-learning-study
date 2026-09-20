#include "inference.h"
#include <regex>
#include <iostream>
#include <algorithm>

#define benchmark
#ifndef min
#define min(a, b) ( ((a) < (b)) ? (a) : (b) )
#endif

YOLO_V8::YOLO_V8() {}
YOLO_V8::~YOLO_V8() {}

template<typename T>
const char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                iBlob[c * imgHeight * imgWidth + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return RET_OK;
}

const char* YOLO_V8::CreateSession(DL_INIT_PARAM& iParams) {
    std::regex pattern("[\u4e00-\u9fa5]");
    bool result = std::regex_search(iParams.modelPath, pattern);
    if (result)
    {
        std::cout << "[YOLO_V8]: Your model path contains invalid characters." << std::endl;
        return "[YOLO_V8]: Create session failed due to path.";
    }
    try
    {
        confThreshold = iParams.confThreshold;
        iouThreshold = iParams.iouThreshold;
        imgSize = iParams.imgSize;
        modelType = iParams.modelType;
        cudaEnable = iParams.cudaEnable;
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOV8");
        Ort::SessionOptions sessionOptions;
        
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOptions.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        sessionOptions.SetLogSeverityLevel(iParams.logSeverityLevel);

        const char* modelPath = iParams.modelPath.c_str();

        session = new Ort::Session(env, modelPath, sessionOptions);
        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputNodesNum = session->GetInputCount();
        for (size_t i = 0; i < inputNodesNum; i++)
        {
            Ort::AllocatedStringPtr inputNodeName = session->GetInputNameAllocated(i, allocator);
            inputNodeNames.emplace_back(inputNodeName.get());
            inputNodeNamesCStr.push_back(inputNodeNames.back().c_str());
        }

        size_t outputNodesNum = session->GetOutputCount();
        for (size_t i = 0; i < outputNodesNum; i++)
        {
            Ort::AllocatedStringPtr outputNodeName = session->GetOutputNameAllocated(i, allocator);
            outputNodeNames.emplace_back(outputNodeName.get());
            outputNodeNamesCStr.push_back(outputNodeNames.back().c_str());
        }
        options = Ort::RunOptions{ nullptr };
        WarmUpSession();
        return RET_OK;
    }
    catch(const std::exception& e)
    {
        std::cout << "[YOLO_V8]: " << e.what() << std::endl;
        return "[YOLO_V8]: Create session failed.";
    }
    
}

const char* YOLO_V8::WarmUpSession() {
    clock_t starttime_1 = clock();
    cv::Mat iImg = cv::Mat(cv::Size(imgSize[0], imgSize[1]), CV_8UC3);
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    if (modelType < 4)
    {
        float* blob = new float[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> YOLO_input_node_dims = { 1, 3, imgSize[0], imgSize[1] };
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), 
            blob, 
            3 * imgSize[0] * imgSize[1], 
            YOLO_input_node_dims.data(), 
            YOLO_input_node_dims.size()
        );
        auto output_tensors = session->Run(options, 
                                           inputNodeNamesCStr.data(), 
                                           &input_tensor, 
                                           inputNodeNamesCStr.size(), 
                                           outputNodeNamesCStr.data(), 
                                           outputNodeNamesCStr.size());
        delete[] blob;
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable) {
            std::cout << "[YOLO_V8(CUDA)]: Cuda warm-up cost " << post_process_time << " ms." << std::endl;
        }
        
    }

    return RET_OK;
}

const char* YOLO_V8::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg) {
    if (iImg.channels() == 3) {
        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    } else {
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
    }

    switch (modelType)
    {
        case YOLO_DETECT:
        case YOLO_POSE:
        case YOLO_DETECT_HALF:
        case YOLO_POSE_HALF: // iImg height(rows): 320, width(cols): 480
        {
            if (iImg.cols >= iImg.rows) {
                resizeScales = iImg.cols / static_cast<float>(iImgSize[0]); // 0.75
                // resize 후 크기: height(rows): 426, width(cols): 640
                cv::resize(oImg, oImg, cv::Size(iImgSize[0], static_cast<int>(iImg.rows / resizeScales)));
            } else {
                resizeScales = iImg.rows / static_cast<float>(iImgSize[1]);
                cv::resize(oImg, oImg, cv::Size(static_cast<int>(iImg.cols / resizeScales), iImgSize[1]));
            }
            cv::Mat tempImg = cv::Mat::zeros(iImgSize[0], iImgSize[1], CV_32FC3);
            oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
            oImg = tempImg;
            break;
        }
    }

    return RET_OK;
}

const char* YOLO_V8::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult) {
#ifdef benchmark
    clock_t starttime_1 = clock();
#endif

    const char* Ret = RET_OK;
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    if (modelType < 4) {
        float* blob = new float[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize[0], imgSize[1] };
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
    }

    return Ret;
}

template<typename N>
const char* YOLO_V8::TensorProcess(clock_t starttime_1, 
                                   cv::Mat& iImg, 
                                   N& blob, 
                                   std::vector<int64_t>& inputNodeDims, 
                                   std::vector<DL_RESULT>& oResult) {
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
        Ort::MemroyInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), 
        blob, 
        3 * imgSize[0] * imgSize[1], 
        inputNodeDims.data(), 
        inputNodeDims.size()
    );
#ifdef benchmark
    clock_t starttime_2 = clock();
#endif
    auto outputTensor = session->Run(options, 
                                     inputNodeNamesCStr.data(), 
                                     &inputTensor, 
                                     inputNodeNamesCStr.size(), 
                                     outputNodeNamesCStr.data(), 
                                     outputNodeNamesCStr.size());
#ifdef benchmark
    clock_t starttime_3 = clock();
#endif
    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<N>::type>();
    delete[] blob;
    switch (modelType)
    {
        case YOLO_DETECT:
        case YOLO_DETECT_HALF:
        {
            int strideNum = static_cast<int>(outputNodeDims[1]);
            int signalResultNum = static_cast<int>(outputNodeDims[2]);
        }
    }
}