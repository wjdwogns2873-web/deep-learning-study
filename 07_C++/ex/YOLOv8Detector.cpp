#include "YOLOv8Detector.hpp"

YOLOv8Detector::YOLOv8Detector(const std::string& modelPath, float confThresh, float nmsThresh)
    : env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8Class"), 
      memoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)), 
      confThreshold(confThresh), 
      nmsThreshold(nmsThresh) {
    
    // M4 CPU 스레드 최적화 설정
    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetInterOpNumThreads(2);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // ONNX 세션 생성 (smart pointer 활용 안전 관리)
    session = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);
    std::cout << "YOLOv8Detector 모델 로드 성공: " << modelPath << std::endl;
}

Ort::Value YOLOv8Detector::preprocess(const cv::Mat& frame, std::vector<float>& inputTensorValues) {
    cv::Mat resizedImage, rgbImage, floatImage;
    cv::resize(frame, resizedImage, cv::Size(inputWidth, inputHeight));
    cv::cvtColor(resizedImage, rgbImage, cv::COLOR_BGR2RGB);
    rgbImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

    inputTensorValues.resize(1 * 3 * inputWidth * inputHeight);
    std::vector<cv::Mat> chwChannels(3);
    for (int i = 0; i < 3; ++i) {
        chwChannels[i] = cv::Mat(inputHeight, inputWidth, CV_32F, inputTensorValues.data() + i * inputHeight * inputWidth);
    }
    cv::split(floatImage, chwChannels);

    return Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size()
    );
}

std::vector<Detection> YOLOv8Detector::postprocess(const cv::Mat& frame, float* floatArray) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    float rx = static_cast<float>(frame.cols) / static_cast<float>(inputWidth);
    float ry = static_cast<float>(frame.rows) / static_cast<float>(inputHeight);

    for (int i = 0; i < numAnchors; ++i) {
        float maxScore = -1.0f;
        int maxClassId = -1;

        for (int c = 0; c < numClasses; ++c) {
            float score = floatArray[(4 + c) * numAnchors + i];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }

        if (maxScore >= confThreshold) {
            float cx = floatArray[0 * numAnchors + i];
            float cy = floatArray[1 * numAnchors + i];
            float bw = floatArray[2 * numAnchors + i];
            float bh = floatArray[3 * numAnchors + i];

            int xmin = static_cast<int>( (cx - bw / 2.0f) * rx );
            int ymin = static_cast<int>( (cy - bh / 2.0f) * ry );
            int boxW = static_cast<int>(bw * rx);
            int boxH = static_cast<int>(bh * ry);

            boxes.push_back(cv::Rect(xmin, ymin, boxW, boxH));
            confidences.push_back(maxScore);
            classIds.push_back(maxClassId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    std::vector<Detection> detections;
    for (int idx : indices) {
        Detection det;
        det.box = boxes[idx];
        det.confidence = confidences[idx];
        det.classId = classIds[idx];
        detections.push_back(det);
    }

    return detections;
}

std::vector<Detection> YOLOv8Detector::detect(const cv::Mat& frame) {
    std::vector<float> inputTensorValues;
    Ort::Value inputTensor = preprocess(frame, inputTensorValues);

    auto outputTensors = session->Run(
        Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1
    );

    float* floatArray = outputTensors[0].GetTensorMutableData<float>();
    return postprocess(frame, floatArray);
}

void YOLOv8Detector::drawResults(cv::Mat& frame, const std::vector<Detection>& detections, float fps) {
    for (const auto& det : detections) {
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);

        std::string label = "Class " + std::to_string(det.classId) + ": " + cv::format("%.2f", det.confidence);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        cv::rectangle(frame, cv::Point(det.box.x, det.box.y - labelSize.height - 5), 
                             cv::Point(det.box.x + labelSize.width, det.box.y), 
                            cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        
    }

    if (fps > 0.0f) {
        std::string fpsText = cv::format("FPS: %.1f", fps);
        cv::putText(frame, fpsText, cv::Point(20, 50), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }
}