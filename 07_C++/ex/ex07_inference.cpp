#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <algorithm>

int main() {
    std::string modelPath = "best.FP32.onnx";
    std::string imagePath = "fruit_0002.png";

    cv::Mat frame = cv::imread(imagePath);
    if (frame.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다: " << imagePath << std::endl;
        return -1;
    }

    cv::Mat resizedImage, rgbImage, floatImage;
    cv::resize(frame, resizedImage, cv::Size(640, 640));
    cv::cvtColor(resizedImage, rgbImage, cv::COLOR_BGR2RGB);
    rgbImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

    int channels = 3;
    int height = 640;
    int width = 640;
    std::vector<float> inputTensorValues(1 * channels * height * width);

    std::vector<cv::Mat> chwChannels(channels);
    for (int i = 0; i < channels; ++i) {
        chwChannels[i] = cv::Mat(height, width, CV_32F, inputTensorValues.data() + i * height * width);
    }
    cv::split(floatImage, chwChannels);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO_Inference");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);

    Ort::Session session(env, modelPath.c_str(), sessionOptions);
    std::cout << "ONNX 모델 로드 성공: " << modelPath << std::endl;

    std:: vector<int64_t> inputShape = {1, channels, height, width};

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
    );

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, 
        inputTensorValues.data(), 
        inputTensorValues.size(), 
        inputShape.data(), 
        inputShape.size()
    );

    const char* inputNames[] = {"images"}; // YOLO 모델 입출력 노드 이름
    const char* outputNames[] = {"output0"};

    // 실제 C++ ONNX 모델 추론 실행
    auto outputTensors = session.Run(
        Ort::RunOptions{nullptr}, 
        inputNames, 
        &inputTensor, 
        1, 
        outputNames, 
        1
    );

    std::cout << "ONNX 모델 C++ 추론 성공!" << std::endl;

    // 추론 결과 정보 확인
    float* floatArray = outputTensors[0].GetTensorMutableData<float>();
    auto typeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    auto outputShape = typeInfo.GetShape();

    std::cout << "출력 텐서 Shape: [";
    for (size_t i = 0; i < outputShape.size(); ++i) {
        std::cout << outputShape[i] << (i < outputShape.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;


    // 후처리에 필요한 변수 선언
    float confThreshold = 0.2f;
    float nmsThreshold = 0.45f;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    int numAnchors = 8400;
    int numClasses = 3;

    float rx = static_cast<float>(frame.cols) / 640.0f;
    float ry = static_cast<float>(frame.rows) / 640.0f;

    for (int i = 0; i < numAnchors; ++i) {
        // 3개 클래스 중 최고 점수 찾기
        float maxScore = -1.0f;
        int maxClassId = -1;

        for (int c = 0; c < numClasses; ++c) {
            // Class 점수는 4번째 인덱스부터 시작
            float score = floatArray[(4 + c) * numAnchors + i];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }

        // 설정한 임계값보다 높은 박스만 수집
        if (maxScore >= confThreshold) {
            float cx = floatArray[0 * numAnchors + i];
            float cy = floatArray[1 * numAnchors + i];
            float bw = floatArray[2 * numAnchors + i];
            float bh = floatArray[3 * numAnchors + i];

            int xmin = static_cast<int>((cx - bw / 2.0f) * rx);
            int ymin = static_cast<int>((cy - bh / 2.0f) * ry);
            int width = static_cast<int>(bw * rx);
            int height = static_cast<int>(bh * ry);

            boxes.push_back(cv::Rect(xmin, ymin, width, height));
            confidences.push_back(maxScore);
            classIds.push_back(maxClassId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    std::cout << "NMS 전 임계값 통과 박스 수: " << boxes.size() << "개" << std::endl;
    std::cout << "최종 검출된 객체 수: " << indices.size() << "개" << std::endl;

    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        int classId = classIds[idx];
        float conf = confidences[idx];

        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);

        std::string label = "Class " + std::to_string(classId) + ": " + cv::format("%.2f", conf);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        cv::rectangle(frame, cv::Point(box.x, box.y - labelSize.height - 5), 
                             cv::Point(box.x + labelSize.width, box.y), 
                             cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, label, cv::Point(box.x, box.y - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    cv::imwrite("output_result.jpg", frame);
    std::cout << "최종 검출된 객체 수: " << indices.size() << "개" << std::endl;

    cv::imshow("YOLO Detection Result", frame);
    cv::waitKey(0);

    return 0;
}