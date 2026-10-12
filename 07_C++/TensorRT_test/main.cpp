#include "YoloTRT.h"
#include <chrono>
#include <numeric>
#include <vector>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

int main() {
    // 1. C++ TensorRT 엔진 로드
    YoloTRT detector("yolov8n.engine");

    // 2. 비디오 파일 열기
    std::string videoPath = "./test_videos/sample.mp4";
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "비디오 파일을 열 수 없습니다: " << videoPath << std::endl;
        return -1;
    }

    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double inputFps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "입력 비디오 정보: " << totalFrames << " Frames, " << inputFps << " FPS" << std::endl;

    // 3. Warm-up (첫 프레임 기반 50회 예열)
    cv::Mat frame;
    cap >> frame;
    if (!frame.empty()) {
        for (int i = 0; i < 50; ++i) {
            detector.detect(frame, 0.25f, 0.45f);
        }
    }

    // 비디오 읽기 위치를 처음 프레임으로 리셋
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    std::vector<double> latencies;
    latencies.reserve(totalFrames);
    int frameCount = 0;

    // 4. 동영상 프레임 단위 End-to-End 측정 (디코딩 + 전처리 + H2D + 추론 + D2H + NMS + 시각화)
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        auto start = std::chrono::high_resolution_clock::now();

        // 전처리 -> TRT 추론 -> NMS 후처리
        auto detections = detector.detect(frame, 0.25f, 0.45f);

        // 검출 결과 바운딩 박스 렌더링
        for (const auto& det : detections) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(ms);
        frameCount++;
    }

    cap.release();

    if (latencies.empty()) {
        std::cerr << "처리된 프레임이 없습니다." << std::endl;
        return -1;
    }

    // 5. 통계 산출
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();
    double min_val = *std::min_element(latencies.begin(), latencies.end());
    double max_val = *std::max_element(latencies.begin(), latencies.end());

    std::cout << "\n================ [5. C++ Native TRT Video 벤치마크] ================" << std::endl;
    std::cout << "처리 프레임 수: " << frameCount << " / " << totalFrames << std::endl;
    std::cout << "평균 Latency (Avg): " << cv::format("%.2f", avg) << " ms" << std::endl;
    std::cout << "최소 Latency (Min): " << cv::format("%.2f", min_val) << " ms" << std::endl;
    std::cout << "최대 Latency (Max): " << cv::format("%.2f", max_val) << " ms" << std::endl;
    std::cout << "평균 동영상 처리 FPS: " << cv::format("%.1f", 1000.0 / avg) << " FPS" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    return 0;
}