#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// 바운딩 박스 구조체
struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

// 이미지 전처리 (640x640 Resize & NCHW 정규화)
std::vector<float> preprocessImage(const cv::Mat& srcImg, int targetW, int targetH) {
    cv::Mat resizedImg;
    cv::resize(srcImg, resizedImg, cv::Size(targetW, targetH));
    cv::cvtColor(resizedImg, resizedImg, cv::COLOR_BGR2RGB);

    resizedImg.convertTo(resizedImg, CV_32FC3, 1.0 / 255.0);

    std::vector<float> inputData(1 * 3 * targetH * targetW);
    int channelSize = targetH * targetW;
    
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; ++i) {
        channels[i] = cv::Mat(targetH, targetW, CV_32FC1, inputData.data() + i * channelSize);
    }
    cv::split(resizedImg, channels);

    return inputData;
}

// YOLOv8 후처리 (Parsing + NMS)
std::vector<Detection> postprocess(const std::vector<float>& outputHost, float confThreshold, float nmsThreshold, int imgWidth, int imgHeight) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    // 출력 Tensor: [1, 84, 8400]
    // 8400개의 박스를 순회하며 각 84개 요소 파싱
    for (int i = 0; i < 8400; ++i) {
        float cx = outputHost[0 * 8400 + i];
        float cy = outputHost[1 * 8400 + i];
        float w  = outputHost[2 * 8400 + i];
        float h  = outputHost[3 * 8400 + i];

        // 80개 클래스 점수 중 최고점 찾기
        float maxScore = 0.0f;
        int maxClassId = -1;
        for (int c = 0; c < 80; ++c) {
            float score = outputHost[(4 + c) * 8400 + i];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }

        if (maxScore >= confThreshold) {
            // 640x640 스케일을 원본 이미지 비율로 복원
            float x1 = (cx - 0.5f * w) * (imgWidth / 640.0f);
            float y1 = (cy - 0.5f * h) * (imgHeight / 640.0f);
            float boxW = w * (imgWidth / 640.0f);
            float boxH = h * (imgHeight / 640.0f);

            boxes.emplace_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1), static_cast<int>(boxW), static_cast<int>(boxH)));
            confidences.emplace_back(maxScore);
            classIds.emplace_back(maxClassId);
        }
    }

    // OpenCV NMS 적용
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    std::vector<Detection> detections;
    detections.reserve(indices.size());
    for (int idx : indices) {
        detections.push_back({ boxes[idx], confidences[idx], classIds[idx] });
    }

    return detections;
}

int main() {
    // 1. 샘플 이미지 다운로드 또는 로드
    std::string imgPath = "bus.jpg";
    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        std::cerr << "이미지 로드 실패! 샘플 이미지를 준비해 주세요." << std::endl;
        return -1;
    }

    // 2. TensorRT Engine 로드
    std::ifstream file("yolov8n.engine", std::ios::binary);
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // 3. 전처리 및 메모리 준비
    std::vector<float> inputHost = preprocessImage(img, 640, 640);
    size_t inputSize = 1 * 3 * 640 * 640 * sizeof(float);
    size_t outputSize = 1 * 84 * 8400 * sizeof(float);
    std::vector<float> outputHost(84 * 8400);

    void* d_input = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    // 4. 추론 실행
    cudaMemcpy(d_input, inputHost.data(), inputSize, cudaMemcpyHostToDevice); // Host to device: CPU -> GPU 복사
    void* bindings[2] = { d_input, d_output };
    
    context->executeV2(bindings);

    cudaMemcpy(outputHost.data(), d_output, outputSize, cudaMemcpyDeviceToHost); // Device to host: GPU -> CPU 복사

    // 5. NMS 후처리
    auto detections = postprocess(outputHost, 0.25f, 0.45f, img.cols, img.rows);

    std::cout << "검출된 객체 수: " << detections.size() << "개" << std::endl;

    // 6. 이미지에 바운딩 박스 그리기 및 저장
    for (const auto& det : detections) {
        cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = "Class " + std::to_string(det.classId) + ": " + cv::format("%.2f", det.conf);
        cv::putText(img, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    cv::imwrite("output_result.jpg", img);
    std::cout << "결과 이미지가 output_result.jpg 로 저장되었습니다!" << std::endl;

    // 7. 메모리 해제
    cudaFree(d_input);
    cudaFree(d_output);
    delete context;
    delete engine;
    delete runtime;

    return 0;
}