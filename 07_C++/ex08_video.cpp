#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <thread>
#include <mutex>
#include <queue>

// 스레드 간 프레임 공유를 위한 스레드 세이프 큐 구조
std::queue<cv::Mat> frameQueue;
std::mutex queueMutex;
bool isRunning = true;

// 프레임 수집 스레드
void captureThreadFunc(std::string videoPath) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) return;

    // 원본 동영상의 FPS 및 프레임당 지연 시간 계산
    double videoFps = cap.get(cv::CAP_PROP_FPS);
    if (videoFps <= 0) videoFps = 30.0; // 예외처리 (기본 30fps)

    int frameDelayMs = static_cast<int>(1000.0 / videoFps);

    cv::Mat frame;
    while (isRunning) {
        auto startTime = std::chrono::high_resolution_clock::now();

        cap >> frame;
        if (frame.empty()) {
            isRunning = false;
            break;
        }

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            // 큐 크기를 1개로 유지 (오래된 프레임 바로 버림으로써 딜레이 제거)
            while (!frameQueue.empty()) {
                frameQueue.pop();
            }
            frameQueue.push(frame.clone());
        }

        // 동영상 원본 FPS 속도에 정확히 맞춰 스레드 대기
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - startTime
        ).count();

        int sleepTime = frameDelayMs - static_cast<int>(elapsed);

        if (sleepTime > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
        }

    }
    cap.release();
}

int main() {
    std::string modelPath = "best.FP32.onnx";

    std::string videoPath = "apple_video.mp4";

    // 캡처 스레드 시작
    std::thread capThread(captureThreadFunc, videoPath);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO_Async");
    Ort::SessionOptions sessionOptions;

    // 단일 연산(Op) 내에서 사용할 CPU 스레드 수 (M4 성능 코어 활용)
    sessionOptions.SetIntraOpNumThreads(4); // CPU 코어 활용 설정

    // 연산 간 병렬 처리를 위한 스레드 수 설정
    sessionOptions.SetInterOpNumThreads(2);

    // CPU 최적화 그래프 레벨 설정 (기본 연산자 융합 최적화)
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // (MPS/CoreML 가속 설정)
    Ort::Session session(env, modelPath.c_str(), sessionOptions);
    std::cout << "ONNX 모델 로드 성공: " << modelPath << std::endl;

    const char* inputNames[] = {"images"};
    const char* outputNames[] = {"output0"};

    cv::Mat currentFrame;
    auto lastTime = std::chrono::high_resolution_clock::now();
    int channels = 3;
    int height = 640;
    int width = 640;
    std::vector<int64_t> inputShape = {1, channels, height, width};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
    );

    float confThreshold = 0.25f;
    float nmsThreshold = 0.45f;
    int numAnchors = 8400;
    int numClasses = 3;

    float fps = 0.0f;

    while (isRunning) {
        // 큐에서 가장 최신 프레임 1장만 꺼내오기
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (frameQueue.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            currentFrame = frameQueue.front();
            frameQueue.pop();
        }


        // Preprocessing (전처리)
        cv::Mat resizedImage, rgbImage, floatImage;
        cv::resize(currentFrame, resizedImage, cv::Size(width, height));
        cv::cvtColor(resizedImage, rgbImage, cv::COLOR_BGR2RGB);
        rgbImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

        std::vector<float> inputTensorValues(1 * channels * height * width);
        std::vector<cv::Mat> chwChannels(channels);
        for (int i = 0; i < channels; ++i) {
            chwChannels[i] = cv::Mat(height, width, CV_32F, inputTensorValues.data() + i * height * width);
        }
        cv::split(floatImage, chwChannels);

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, 
            inputTensorValues.data(), 
            inputTensorValues.size(), 
            inputShape.data(), 
            inputShape.size()
        );

        // Inference (추론)
        auto outputTensors = session.Run(
            Ort::RunOptions{nullptr}, 
            inputNames, 
            &inputTensor, 
            1, 
            outputNames, 
            1
        );

        float* floatArray = outputTensors[0].GetTensorMutableData<float>();

        // Postprocessing & Parsing (후처리 및 NMS)
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;

        float rx = static_cast<float>(currentFrame.cols) / static_cast<float>(width);
        float ry = static_cast<float>(currentFrame.rows) / static_cast<float>(height);

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

        // Visualization & FPS Render (시각화)
        for (int idx : indices) {
            cv::Rect box = boxes[idx];
            int classId = classIds[idx];
            float conf = confidences[idx];

            cv::rectangle(currentFrame, box, cv::Scalar(0, 255, 0), 2);

            std::string label = "Class: " + std::to_string(classId) + ": " + cv::format("%.2f", conf);
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            cv::rectangle(currentFrame, cv::Point(box.x, box.y - labelSize.height - 5), 
                                 cv::Point(box.x + labelSize.width, box.y), 
                                 cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(currentFrame, label, cv::Point(box.x, box.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        // FPS
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = currentTime - lastTime;
        lastTime = currentTime;
        // std::cout << "duration: " << duration.count() << std::endl; // 약 0.0653474

        // 화면 우상단에 FPS 표시
        std::string fpsText = cv::format("FPS: %.1f", 1.0f / duration.count());
        cv::putText(currentFrame, fpsText, cv::Point(20, 40), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        
        cv::imshow("YOLO Real-Time Detection", currentFrame);

        // 'q' 키 또는 ESC 키 누르면 종료 (1ms 대기)
        if (cv::waitKey(1) == 'q' || cv::waitKey(1) == 27) {
            isRunning = false;
            break;
        }
    }

    if (capThread.joinable()) {
        capThread.join(); // 스레드 안전 종료
    }

    cv::destroyAllWindows();
    return 0;
}