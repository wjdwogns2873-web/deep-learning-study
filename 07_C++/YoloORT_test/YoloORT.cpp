#include "YoloORT.hpp"
#if defined(__APPLE__)
#include <coreml_provider_factory.h> // ⭐ CoreML 헤더 필수 포함!
#endif

YoloORT::YoloORT(const std::string& modelPath, bool isFP16, const cv::Size& inputSize)
    : env(ORT_LOGGING_LEVEL_WARNING, "YoloORT"), isFP16Mode(isFP16), inputImageSize(inputSize) {

    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // M4 Mac 호스트 환경일 경우 CoreML EP 사용 가능
#if defined(__APPLE__)
    uint32_t coreml_flags = 0;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(
        static_cast<OrtSessionOptions*>(sessionOptions), coreml_flags));
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

    LetterboxInfo info;

    if (isFP16Mode) {
        std::vector<Ort::Float16_t> inputTensorValues(inputTensorSize);
        info = preprocessFP16(frame, inputTensorValues);

        inputTensors.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
            memoryInfo, inputTensorValues.data(), inputTensorSize, inputShape.data(), inputShape.size()));
    } else {
        std::vector<float> inputTensorValues(inputTensorSize);
        info = preprocess(frame, inputTensorValues);

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
    std::vector<Detection> detections = postprocess(outputTensors, info, confThreshold, nmsThreshold);
    return detections;
}

LetterboxInfo YoloORT::preprocess(const cv::Mat& frame, std::vector<float>& inputTensorValues) {
    auto [letterbox_img, info] = letterbox(frame);

    cv::cvtColor(letterbox_img, letterbox_img, cv::COLOR_BGR2RGB);
    letterbox_img.convertTo(letterbox_img, CV_32FC3, 1.0 / 255.0);

    // HWC -> CHN (Zero copy 기반 채널 분리)
    int channelSize = inputImageSize.width * inputImageSize.height;
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; i++) {
        channels[i] = cv::Mat(inputImageSize.height, inputImageSize.width, CV_32FC1, inputTensorValues.data() + i * channelSize);
    }
    cv::split(letterbox_img, channels);

    return info;
}

LetterboxInfo YoloORT::preprocessFP16(const cv::Mat& frame, std::vector<Ort::Float16_t>& inputTensorValues) {
    // FP32 전처리 후 Ort::Float16_t 변환 매핑
    std::vector<float> fp32Values(inputTensorValues.size());
    LetterboxInfo info = preprocess(frame, fp32Values);

    for (size_t i = 0; i < fp32Values.size(); i++) {
        inputTensorValues[i] = Ort::Float16_t(fp32Values[i]);
    }

    return info;
}

std::vector<Detection> YoloORT::postprocess(std::vector<Ort::Value>& outputTensors, const LetterboxInfo& info, float confThreshold, float nmsThreshold) {
    // float* data = outputTensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape(); // [1, 84, 8400]

    int classCount = static_cast<int>(outputShape[1] - 4);
    int numAnchors = static_cast<int>(outputShape[2]);

    std::vector<float> floatData(outputShape[1] * numAnchors);

    if (isFP16Mode) {
        auto* fp16Data = outputTensors[0].GetTensorMutableData<Ort::Float16_t>();
        for (size_t i = 0; i < floatData.size(); ++i) {
            floatData[i] = static_cast<float>(fp16Data[i]);
        }
    } else {
        auto* fp32Data = outputTensors[0].GetTensorMutableData<float>();
        std::copy(fp32Data, fp32Data + floatData.size(), floatData.begin());
    }

    float* data = floatData.data();

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

            // 레터박스 적용했던거 원본으로 복원
            int realX1 = static_cast<int>((x1 - info.padW) / info.scale);
            int realY1 = static_cast<int>((y1 - info.padH) / info.scale);
            int realW = static_cast<int>(width / info.scale);
            int realH = static_cast<int>(height / info.scale);

            // 원본이미지와의 교집합을 하게 되면 복잡한 min, max 클리핑이 필요없음!
            // 원본 이미지 경계를 벗어나지 않도록 클리핑(Clipping)
            // realX1 = std::max(0, std::min(realX1, info.orgImgSize.width));
            // realY1 = std::max(0, std::min(realY1, info.orgImgSize.height));

            cv::Rect box(realX1, realY1, realW, realH);
            cv::Rect bounds(0, 0, info.orgImgSize.width, info.orgImgSize.height);
            box &= bounds; // 원본 이미지와의 교집합 구하기.(xmin + width가 전체 width를 넘어가는걸 방지)

            if (box.width > 0 && box.height > 0) {
                boxes.push_back(box);
                confidences.push_back(maxScore);
                classIds.push_back(maxClassId);
            }
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

std::pair<cv::Mat, LetterboxInfo> YoloORT::letterbox(const cv::Mat& src, int targetW, int targetH) {
    LetterboxInfo info;
    info.orgImgSize = cv::Size(src.cols, src.rows);

    info.scale = std::min(static_cast<float>(targetW) / src.cols, static_cast<float>(targetH) / src.rows);

    int newW = static_cast<int>(std::round(src.cols * info.scale));
    int newH = static_cast<int>(std::round(src.rows * info.scale));

    info.padW = (targetW - newW) / 2; // 좌우 패딩 너비
    info.padH = (targetH - newH) / 2; // 위아래 패딩 높이

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH));

    int top = info.padH;
    int bottom = targetH - newH - info.padH;
    int left = info.padW;
    int right = targetW - newW - info.padW;

    cv::Mat letterbox_img;
    cv::copyMakeBorder(resized, letterbox_img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return {letterbox_img, info};
}