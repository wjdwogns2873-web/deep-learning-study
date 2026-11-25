#include "YoloTRT.h"
#include <chrono>
#include <numeric>
#include <vector>
#include <iostream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <memory>
#include <iomanip>
#include <ctime>
#include <opencv2/opencv.hpp>

class LockFreeFrameQueue {
private:
    static constexpr size_t kCapacity = 32;
    std::shared_ptr<cv::Mat> ringBuffer[kCapacity];

    std::atomic<size_t> pushIndex{0};
    std::atomic<size_t> popIndex{0};

public:
    LockFreeFrameQueue() = default;

    // producer: 복사 없이 우측값 참조(&&)로 포인터 소유권만 이동
    bool push(std::shared_ptr<cv::Mat>&& framePtr) {
        size_t currentPush = pushIndex.load(std::memory_order_relaxed);
        size_t currentPop = popIndex.load(std::memory_order_acquire);

        if ( (currentPush - currentPop) >= kCapacity ) {
            return false;
        }

        size_t index = currentPush % kCapacity;
        ringBuffer[index] = std::move(framePtr);

        pushIndex.store(currentPush + 1, std::memory_order_release);
        return true;
    }

    // consumer: 링 버퍼에 있는 포인터 소유권만 소비자 변수로 이동
    bool pop(std::shared_ptr<cv::Mat>& framePtr, const std::atomic<bool>& stopSignal) {
        while (true) {
            size_t currentPop = popIndex.load(std::memory_order_relaxed);
            size_t currentPush = pushIndex.load(std::memory_order_acquire);

            if (currentPop == currentPush) {
                if (stopSignal.load(std::memory_order_relaxed)) {
                    return false;
                }
                std::this_thread::yield();
                continue;
            }

            size_t index = currentPop % kCapacity;
            framePtr = std::move(ringBuffer[index]); // 주소 소유권만 쏙 빼옴

            popIndex.store(currentPop + 1, std::memory_order_release);
            return true;
        }
    }
};

void videoDecodingThread(
    const std::string& videoPath, 
    LockFreeFrameQueue& lfQueue, 
    LockFreeFrameQueue& objectPool, 
    std::atomic<bool>& stopSignal
) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "비디오 파일을 열 수 없습니다: " << videoPath << std::endl;
        stopSignal = true;
        return;
    }

    std::shared_ptr<cv::Mat> framePtr;
    while (!stopSignal) {
        // 오브젝트 풀에서 놀고 있는 빈 방 주소를 꺼냅니다.
        if (!objectPool.pop(framePtr, stopSignal)) {
            break;
        }

        // 꺼내온 기존 메모리 공간에 픽셀만 덮어씁니다.
        cap >> *framePtr;
        if (framePtr->empty()) {
            objectPool.push(std::move(framePtr));
            break;
        }

        // 추론 큐로 포인터 소유권 이동
        while (!lfQueue.push(std::move(framePtr)) && !stopSignal) {
            std::this_thread::yield();
        }
    }
    cap.release();
    stopSignal = true;
}

int main() {
    std::string engineName = "yolov8n.int8.engine";
    YoloTRT detector(engineName);
    
    std::string videoPath = "./test_videos/sample.mp4";

    cv::VideoCapture tempCap(videoPath);
    int totalFrames = static_cast<int>(tempCap.get(cv::CAP_PROP_FRAME_COUNT));
    tempCap.release();

    cv::Mat dummyMat = cv::Mat::zeros(640, 640, CV_8UC3);
    for (int i = 0; i < 50; i++) {
        detector.detect(dummyMat, 0.25f, 0.45f);
    }

    LockFreeFrameQueue lfQueue;
    LockFreeFrameQueue objectPool;

    for (int i = 0; i < 32; i++) {
        auto allocatedMat = std::make_shared<cv::Mat>(640, 640, CV_8UC3);
        objectPool.push(std::move(allocatedMat));
    }


    std::atomic<bool> stopSignal{false};
    std::vector<double> latencies;
    latencies.reserve(totalFrames);

    std::thread reader(videoDecodingThread, videoPath, std::ref(lfQueue), std::ref(objectPool), std::ref(stopSignal));

    int frameCount = 0;
    std::shared_ptr<cv::Mat> framePtr;

    while (true) {
        if (!lfQueue.pop(framePtr, stopSignal)) {
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // 역참조(*framePtr)로 추론 및 박스 그리기 수행
        auto detections = detector.detect(*framePtr, 0.25f, 0.45f);
        for (const auto& det : detections) {
            cv::rectangle(*framePtr, det.box, cv::Scalar(0, 255, 0), 2);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(ms);
        frameCount++;

        // 사용이 끝난 포인터 방을 파괴하지 않고 오브젝트 풀로 반납
        while (!objectPool.push(std::move(framePtr))) {
            std::this_thread::yield();
        }
    }

    if (reader.joinable()) reader.join();


    // 한국 시간에 맞추기 위해 9시간을 더합니다.
    auto now = std::chrono::system_clock::now() + std::chrono::hours(9);
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();
    // double min = *std::min_element(latencies.begin(), latencies.end());
    // double max = *std::max_element(latencies.begin(), latencies.end());

    std::cout << "측정 시각(KST): " << std::put_time(std::gmtime(&now_c), "%Y-%m-%d %H:%M:%S") << std::endl;
    
    std::cout << "사용한 가속 엔진: " << engineName << std::endl;
    // std::cout << "처리 프레임 수: " << frameCount << " / " << totalFrames << std::endl;
    // std::cout << "순수 추론 파이프라인 Latency (Avg): " << cv::format("%.2f", avg) << " ms" << std::endl;
    // std::cout << "최소 Latency (Min): " << cv::format("%.2f", min) << " ms" << std::endl;
    // std::cout << "최대 Latency (Max): " << cv::format("%.2f", max) << " ms" << std::endl;
    std::cout << "2threads 방식에서 Object Pool, 스마트 포인터 사용한 처리 속도: " << cv::format("%.1f", 1000.0 / avg) << " FPS" << std::endl;

    return 0;
}