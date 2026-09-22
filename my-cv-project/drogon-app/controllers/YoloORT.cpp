#include "YoloORT.hpp"

YoloORT::YoloORT(const std::string& modelPath, bool isFP16, const cv::Size& inputSize)
    : env(ORT_LOGGING_LEVEL_WARNING, "YoloORT"), isFP16Mode(isFP16), inputImageSize(inputSize) {

    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // M4 Mac 호스트 환경일 경우 CoreML EP 사용 가능
    #if defined(__APPLE__)
    // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOptions, 0));
    #endif

    session = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    // 입력/출력 텐서 이름 확보 (메모리 관리 유의)
    auto inputNameAlloc = session->GetInputNameAllocated(0, allocator);
    inputNames.push_back(inputNameAlloc.get());
    inputNamePtrs.push_back(inputNames.back().c_str());

    auto outputNameAlloc = session->GetOutputNameAllocated(0, allocator);
    outputNames.push_back(outputNameAlloc.get());
    outputNamePtrs.push_back(outputNames.back().c_str());

    inputShape = {1, 3, inputImageSize.height, inputImageSize.width};
}

std::vector<Detection> YoloORT::detect(const cv::Mat& frame, float confThreshold, float nmsThreshold) {
    int inputTensorSize = 1 * 3 * inputImageSize.height * inputImageSize.width;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<Ort::Value> inputTensors;

    if (isFP16Mode) {
        std::vector<Ort::Float16_t> inputTensorValues(inputTensorSize);
        preprocessFP16(frame, inputTensorValues);

        inputTensors.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
            memoryInfo, inputTensorValues.data(), inputTensorSize, inputShape.data(), inputShape.size()));
    } else {
        std::vector<float> inputTensorValues(inputTensorSize);
        preprocess(frame, inputTensorValues);

        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize, inputShape.data(), inputShape.size()));
    }

    // ONNX Runtime C++ 비동기/동기 추론
    auto outputTensors = session->Run(
        Ort::RunOptions{nullptr}, 
        inputNamePtrs.data(), 
        inputTensors.data(), 
        inputNamePtrs.size(), 
        outputNamePtrs.data(), 
        outputNamePtrs.size()
    );

    // 후처리(NMS) 진행 후 Detection 결과 반환
    std::vector<Detection> detections = postprocess(outputTensors, confThreshold, nmsThreshold);
    return detections;
}

void YoloORT::preprocess(const cv::Mat& frame, std::vector<float>& inputTensorValues) {
    cv::Mat resized;
    cv::resize(frame, resized, inputImageSize);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    // HWC -> CHN (Zero copy 기반 채널 분리)
    int channelSize = inputImageSize.width * inputImageSize.height;
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; i++) {
        channels[i] = cv::Mat(inputImageSize.height, inputImageSize.width, CV_32FC1, inputTensorValues.data() + i * channelSize);
    }
    cv::split(resized, channels);
}

void YoloORT::preprocessFP16(const cv::Mat& frame, std::vector<Ort::Float16_t>& inputTensorValues) {
    // FP32 전처리 후 Ort::Float16_t 변환 매핑
    std::vector<float> fp32Values(inputTensorValues.size());
    preprocess(frame, fp32Values);

    for (size_t i = 0; i < fp32Values.size(); i++) {
        inputTensorValues[i] = Ort::Float16_t(fp32Values[i]);
    }
}

std::vector<Detection> YoloORT::postprocess(std::vector<Ort::Value>& outputTensors, float confThreshold, float nmsThreshold) {
    float* data = outputTensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape(); // [1, 84, 8400]

    int classCount = static_cast<int>(outputShape[1] - 4);
    int numAnchors = static_cast<int>(outputShape[2]);

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < numAnchors; i++) {

        float maxScore = 0.0f;
        int maxClassId = -1;
        for (int c = 0; c < classCount; c++) {
            float score = data[(4 + c) * numAnchors + i];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }

        if (maxScore > confThreshold) {
            float cx = data[0 * numAnchors + i];
            float cy = data[1 * numAnchors + i];
            float w = data[2 * numAnchors + i];
            float h = data[3 * numAnchors + i];

            int x1 = static_cast<int>(cx - w / 2.0f);
            int y1 = static_cast<int>(cy - h / 2.0f);
            int width = static_cast<int>(w);
            int height = static_cast<int>(h);

            cv::Rect box(x1, y1, w, h);
            boxes.push_back(box);
            confidences.push_back(maxScore);
            classIds.push_back(maxClassId);
        }

    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    std::vector<Detection> detections;
    detections.reserve(indices.size());
    for (int idx : indices) {
        detections.push_back({boxes[idx], confidences[idx], classIds[idx]});
    }

    return detections;
}