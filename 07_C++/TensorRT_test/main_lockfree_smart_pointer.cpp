#include "YoloTRT.h"
#include <chrono>
#include <numeric>
#include <vector>
#include <iostream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <memory> // shared_ptr 사용을 위해 추가
#include <opencv2/opencv.hpp>

// 1. 큐B에서 주고받을 데이터 구조체 (frame을 스마트 포인터로 변경!)
struct InferenceResult {
    std::shared_ptr<cv::Mat> frame;
    std::vector<Detection> detections;
};

// 2. 주소값(포인터)만 원자적으로 토스하는 초고속 Lock-Free 큐
template <typename T>
class LockFreeQueue {
private:
    static constexpr size_t kCapacity = 32;
    T ringBuffer[kCapacity]; // 이제 이 배열은 Mat 객체가 아니라 8바이트 포인터(또는 가벼운 구조체)만 담습니다.

    std::atomic<size_t> pushIndex{0};
    std::atomic<size_t> popIndex{0};

public:
    LockFreeQueue() {
        // [중요] 포인터나 무브 전용 구조체를 담으므로, 생성자에서 대용량 zeros() 공간을 미리 파둘 필요가 완전히 사라집니다!
    }

    // producer: 소유권을 링 버퍼로 완전히 쓱 밀어 넣음 (카피 발생 X)
    bool push(T&& item, std::atomic<bool>& stopSignal) {
        size_t currentPush = pushIndex.load(std::memory_order_relaxed);
        size_t currentPop = popIndex.load(std::memory_order_acquire);

        if ((currentPush - currentPop) >= kCapacity) {
            return false;
        }

        size_t index = currentPush % kCapacity;
        
        // 무거운 copyTo를 전부 걷어내고, 오직 포인터 주소/소유권만 이동(std::move)시킵니다.
        ringBuffer[index] = std::move(item);

        pushIndex.store(currentPush + 1, std::memory_order_release);
        return true;
    }

    // consumer: 링 버퍼에 있던 소유권을 소비자 바구니로 쏙 빼앗아 옴 (카피 발생 X)
    bool pop(T& item, std::atomic<bool>& stopSignal) {
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
            
            // 소비자가 데이터를 가져갈 때도 오직 주소 소유권만 쏙 빼내어 넘겨줍니다.
            item = std::move(ringBuffer[index]);

            popIndex.store(currentPop + 1, std::memory_order_release);
            return true;
        }
    }
};

// ==========================================
// 스마트 포인터 기반의 Zero-Copy 스레드 함수들
// ==========================================

// 스레드 1: 동영상 디코딩 후 스마트 포인터에 담아 큐A에 투하
void videoDecodingThread(
    const std::string& videoPath, 
    LockFreeQueue<std::shared_ptr<cv::Mat>>& lfQueueA, 
    LockFreeQueue<std::shared_ptr<cv::Mat>>& objectPool, // 주입받음
    std::atomic<bool>& stopSignalA
) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "비디오 파일을 열 수 없습니다: " << videoPath << std::endl;
        stopSignalA = true;
        return;
    }

    std::shared_ptr<cv::Mat> framePtr;

    while (!stopSignalA) {
        // ★ [핵심] make_shared를 전면 제거! 오브젝트 풀 큐에서 놀고 있는 방 주소를 꺼내옵니다.
        while (!objectPool.pop(framePtr, stopSignalA)) {
            if (stopSignalA.load()) return;
            std::this_thread::yield();
        }

        cap >> *framePtr; // 꺼내온 기존 메모리 공간에 픽셀값만 그대로 덮어쓰기!
        
        if (framePtr->empty()) {
            // 빈 프레임이면 사용하려던 포인터를 다시 풀에 돌려주고 탈출
            objectPool.push(std::move(framePtr), stopSignalA);
            break;
        }

        while (!lfQueueA.push(std::move(framePtr), stopSignalA)) {
            std::this_thread::yield();
        }
    }
    cap.release();
    stopSignalA = true;
}

// 스레드 2: 주소값만 꺼내서 GPU 추론 후 큐B로 토스
void gpuInferenceThread(
    YoloTRT& detector, 
    LockFreeQueue<std::shared_ptr<cv::Mat>>& lfQueueA, 
    LockFreeQueue<InferenceResult>& lfQueueB, 
    std::atomic<bool>& stopSignalA, 
    std::atomic<bool>& stopSignalB
) {
    std::shared_ptr<cv::Mat> framePtr;

    while (true) {
        // 큐A에서 이미지 주소(shared_ptr)만 쏙 빼옵니다. (카피 부하 0)
        if (!lfQueueA.pop(framePtr, stopSignalA)) {
            break;
        }

        // 역참조(*framePtr)를 통해 실메모리에 접근하여 GPU 추론 수행
        std::vector<Detection> detections = detector.detect(*framePtr, 0.25f, 0.45f);

        InferenceResult res;
        res.frame = std::move(framePtr); // 주소 소유권을 구조체로 이관
        res.detections = std::move(detections);

        // 큐B에 구조체 패키지를 던질 때도 오직 주소와 벡터 포인터만 이동합니다. (카피 부하 0)
        while (!lfQueueB.push(std::move(res), stopSignalB)) {
            std::this_thread::yield();
        }
    }
    stopSignalB = true;
}

// 스레드 3: 주소값만 받아와서 최종 시각화/후처리 연산 수행
void cpuPostProcessingThread(
    LockFreeQueue<InferenceResult>& lfQueueB, 
    LockFreeQueue<std::shared_ptr<cv::Mat>>& objectPool, // 주입받음
    std::atomic<bool>& stopSignalB, 
    int& frameCount
) {
    InferenceResult res;

    while (true) {
        if (!lfQueueB.pop(res, stopSignalB)) {
            break;
        }

        for (const auto& det : res.detections) {
            cv::rectangle(*res.frame, det.box, cv::Scalar(0, 255, 0), 2);
        }

        frameCount++;

        // ★ [핵심] 처리가 끝난 이미지 포인터 주소를 파괴(Free)하지 않고, 오브젝트 풀에 다시 반납(Push)합니다!
        // 이로써 무한히 순환하는 Zero-Allocation 컨베이어 벨트가 완성됩니다.
        std::shared_ptr<cv::Mat> recycledMat = std::move(res.frame);
        while (!objectPool.push(std::move(recycledMat), stopSignalB)) {
            std::this_thread::yield();
        }
    }
}


int main() {
    std::string engineName = "yolov8n.fp32.engine";

    YoloTRT detector(engineName);
    std::string videoPath = "./test_videos/sample.mp4";

    cv::VideoCapture tempCap(videoPath);
    int totalFrames = static_cast<int>(tempCap.get(cv::CAP_PROP_FRAME_COUNT));
    tempCap.release();

    cv::Mat dummyFrame = cv::Mat::zeros(640, 640, CV_8UC3);
    for (int i = 0; i < 50; i++) {
        detector.detect(dummyFrame, 0.25f, 0.45f);
    }

    LockFreeQueue<std::shared_ptr<cv::Mat>> lfQueueA; // 영상 읽기 -> GPU 추론
    LockFreeQueue<InferenceResult> lfQueueB; // GPU 추론 -> 후처리/시각화




    // [추가] 오브젝트 풀 큐 생성: 놀고 있는 빈 포인터 방 32개를 담아둘 큐
    LockFreeQueue<std::shared_ptr<cv::Mat>> objectPool;
    std::atomic<bool> dummySignal{false};

    // 32개의 Mat 공간을 미리 힙에 할당하여 오브젝트 풀에 채워 넣습니다. (동적 할당 전면 배제)
    for (int i = 0; i < 32; i++) {
        auto allocatedMat = std::make_shared<cv::Mat>(640, 640, CV_8UC3);
        objectPool.push(std::move(allocatedMat), dummySignal);
    }


    std::atomic<bool> stopSignalA{false};
    std::atomic<bool> stopSignalB{false};
    std::vector<double> latencies;
    // latencies.reserve(totalFrames);
    int frameCount = 0;

    auto totalStart = std::chrono::high_resolution_clock::now();

    // 스레드 3개 가동
    // 스레드 1: 디코딩
    std::thread reader(videoDecodingThread, videoPath, std::ref(lfQueueA), std::ref(objectPool), std::ref(stopSignalA));

    // 스레드 2: GPU 추론 전용 스레드 가동
    std::thread inferencer(gpuInferenceThread, std::ref(detector), std::ref(lfQueueA), std::ref(lfQueueB), std::ref(stopSignalA), std::ref(stopSignalB));

    // 스레드 3: CPU 후처리 및 시각화 스레드를 메인 스레드가 직접 수행하거나 별도 스레드로 분리
    // 여기서는 메인 스레드에서 직접 돌리거나 별도 스레드로 관리할 수 있습니다.
    std::thread transformer(cpuPostProcessingThread, std::ref(lfQueueB), std::ref(objectPool), std::ref(stopSignalB), std::ref(frameCount));

    // 안전하게 스레드 합류
    if (reader.joinable()) reader.join();
    if (inferencer.joinable()) inferencer.join();
    if (transformer.joinable()) transformer.join();

    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalSec = std::chrono::duration<double>(totalEnd - totalStart).count();

    double realFPS = frameCount / totalSec;

    // 최종 벤치마크 결과 계산 및 출력
    // double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    // double avg = sum / latencies.size();
    // double min = *std::min_element(latencies.begin(), latencies.end());
    // double max = *std::max_element(latencies.begin(), latencies.end());

    std::cout << "[Lock-Free Smart Pointer 벤치마크]" << std::endl;
    std::cout << "가속 엔진: " << engineName << std::endl;
    std::cout << "처리 프레임 수: " << frameCount << " / " << totalFrames << std::endl;
    std::cout << "총 소요 시간: " << cv::format("%.2f", totalSec) << " 초" << std::endl;
    std::cout << "진짜 파이프라인 처리 속도: " << cv::format("%.1f", realFPS) << " FPS" << std::endl;

    return 0;
}