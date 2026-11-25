#include "YoloTRT.h"
#include <chrono>
#include <numeric>
#include <vector>
#include <iostream>
#include <algorithm>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <opencv2/opencv.hpp>

// 쓰레드 간 안전하게 프레임을 주고받기 위한 공유 큐 구조체
struct FrameQueue {
    std::queue<cv::Mat> queue;
    std::mutex mtx;
    std::condition_variable cv;
    const size_t maxSize = 30; // 메모리 폭주 방지를 위한 버퍼 상한선

    void push(const cv::Mat& frame) {
        std::unique_lock<std::mutex> lock(mtx);
        // true일 때는 아래로 내려감. false일 때는 자물쇠를 내려놓고 기다림
        cv.wait(lock, [this]() { return queue.size() < maxSize; });
        queue.push(frame.clone());
        cv.notify_one();
    }

    bool pop(cv::Mat& frame, const std::atomic<bool>& stop) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this, &stop]() { return !queue.empty() || stop; });
        if (queue.empty() && stop) return false;
        frame = queue.front();
        queue.pop();
        cv.notify_one();
        return true;
    }
};

// [쓰레드 1] 동영상 전용 고속 디코딩 쓰레드 루프
void videoDecodingThread(const std::string& videoPath, FrameQueue& rawQueue, std::atomic<bool>& stopSignal) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "❌ 비디오 파일을 열 수 없습니다: " << videoPath << std::endl;
        stopSignal = true;
        return;
    }

    cv::Mat frame;
    while (!stopSignal) {
        cap >> frame;
        if (frame.empty()) break; // 동영상 끝
        rawQueue.push(frame);
    }
    cap.release();
    stopSignal = true; // 디코딩 종료 알림
    rawQueue.cv.notify_all();
}

int main() {
    // 각 모델을 테스트할 때 여기 엔진 파일명만 바꿔주며 실행하세요!
    std::string engineName = "yolov8n.fp32.engine";
    // std::string engineName = "yolov8n.fp16.engine";
    // std::string engineName = "yolov8n.int8.engine";

    std::cout << "[멀티쓰레드 벤치마크] 로드 중인 엔진: " << engineName << std::endl;
    YoloTRT detector(engineName);

    std::string videoPath = "./test_videos/sample.mp4";
    
    // 비디오 전체 프레임 수 미리 파악용 (출력용)
    cv::VideoCapture tempCap(videoPath);
    int totalFrames = static_cast<int>(tempCap.get(cv::CAP_PROP_FRAME_COUNT));
    tempCap.release();

    // 웜업 (예열)
    cv::Mat dummyFrame = cv::Mat::zeros(640, 640, CV_8UC3);
    for (int i = 0; i < 50; ++i) {
        detector.detect(dummyFrame, 0.25f, 0.45f);
    }

    FrameQueue rawQueue;
    std::atomic<bool> stopSignal(false);
    std::vector<double> latencies;
    latencies.reserve(totalFrames > 0 ? totalFrames : 3000);

    // 디코딩 쓰레드 가동 (CPU가 동영상을 무정지로 계속 읽어 큐에 쌓음)
    std::thread reader(videoDecodingThread, videoPath, std::ref(rawQueue), std::ref(stopSignal));

    std::cout << "⚡ 멀티 쓰레드 파이프라인 가동... 추론 측정 시작합니다." << std::endl;
    int frameCount = 0;

    // [쓰레드 2 (Main)] 고속 GPU 추론 및 후처리 렌더링
    while (true) {
        cv::Mat frame;
        // 디코딩 쓰레드가 읽어온 프레임을 비동기적으로 Pop (없으면 대기)
        if (!rawQueue.pop(frame, stopSignal)) {
            break; 
        }

        auto start = std::chrono::high_resolution_clock::now();

        // 1. 전처리 -> TRT 추론 -> NMS 후처리
        auto detections = detector.detect(frame, 0.25f, 0.45f);

        // 2. 검출 결과 바운딩 박스 렌더링
        for (const auto& det : detections) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(ms);
        frameCount++;
    }

    // 쓰레드 안전 종료 및 합류
    if (reader.joinable()) {
        reader.join();
    }

    if (latencies.empty()) {
        std::cerr << "처리된 프레임이 없습니다." << std::endl;
        return -1;
    }

    // 5. 통계 산출
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();
    double min = *std::min_element(latencies.begin(), latencies.end());
    double max = *std::max_element(latencies.begin(), latencies.end());

    std::cout << "\n================ [비동기 멀티쓰레드 TRT Video 벤치마크] ================" << std::endl;
    std::cout << "사용한 가속 엔진: " << engineName << std::endl;
    std::cout << "처리 프레임 수: " << frameCount << " / " << totalFrames << std::endl;
    std::cout << "순수 추론 파이프라인 Latency (Avg): " << cv::format("%.2f", avg) << " ms" << std::endl;
    std::cout << "최소 Latency (Min): " << cv::format("%.2f", min) << " ms" << std::endl;
    std::cout << "최대 Latency (Max): " << cv::format("%.2f", max) << " ms" << std::endl;
    std::cout << "최종 동영상 처리 속도: " << cv::format("%.1f", 1000.0 / avg) << " FPS" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    return 0;
}