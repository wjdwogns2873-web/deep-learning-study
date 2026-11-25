#include "YoloTRT.h"
#include <chrono>
#include <numeric>
#include <vector>
#include <iostream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>

// 큐B(추론 완료 -> NMS/시각화)에서 사용할 데이터 구조체
struct InferenceResult {
    cv::Mat frame;
    std::vector<Detection> detections;
    // std::chrono::high_resolution_clock::time_point startTime;
};

// 2. 범용 Lock-Free 순환 큐 클래스 (템플릿 적용)
template <typename T>
class LockFreeQueue {
private:
    static constexpr size_t kCapacity = 32; // 링 버퍼의 최대 용량
    T ringBuffer[kCapacity]; // 고정 크기 배열 (메모리 동적 할당 방지)

    std::atomic<size_t> pushIndex{0};
    std::atomic<size_t> popIndex{0};

public:
    LockFreeQueue() {
        // cv::Mat 타입일 때만 초기 640 640 방 32개를 미리 파두도록 예외 처리
        if constexpr (std::is_same_v<T, cv::Mat>) {
            for (size_t i = 0; i < kCapacity; i++) {
                ringBuffer[i] = cv::Mat::zeros(640, 640, CV_8UC3);
            }
        } else if constexpr (std::is_same_v<T, InferenceResult>) {
            for (size_t i = 0; i < kCapacity; i++) {
                ringBuffer[i].frame = cv::Mat::zeros(640, 640, CV_8UC3);
            }
        }
    }

    // producer: 큐에 데이터 삽입
    // producer: std::move를 통해 우측값 참조로 소유권만 이관받아 픽셀 복사 원천 차단
    bool push(T&& item, std::atomic<bool>& stopSignal) {
        size_t currentPush = pushIndex.load(std::memory_order_relaxed);
        size_t currentPop = popIndex.load(std::memory_order_acquire);

        // 큐가 가득 찼는지 체크
        if ((currentPush - currentPop) >= kCapacity) {
            return false;
        }

        size_t index = currentPush % kCapacity;

        // cv::Mat일 때는 메모리 풀링을 위해 copyTo 사용, 일반 구조체는 대입 연산 사용
        if constexpr (std::is_same_v<T, cv::Mat>) {
            item.copyTo(ringBuffer[index]);
        } else if constexpr (std::is_same_v<T, InferenceResult>) {
            // 이미 방이 만들어져 있으므로 copyTo를 명시하여 픽셀 복사 오버헤드를 막고 메모리 풀을 활용함.
            item.frame.copyTo(ringBuffer[index].frame);
            ringBuffer[index].detections = std::move(item.detections); // 벡터는 소유권 째로 가볍게 이동
        } else {
            ringBuffer[index] = std::move(item);
        }

        pushIndex.store(currentPush + 1, std::memory_order_release);
        return true;
    }

    // consumer: 큐에서 데이터 추출
    bool pop(T& item, std::atomic<bool>& stopSignal) {
        while (true) {
            size_t currentPop = popIndex.load(std::memory_order_relaxed);
            size_t currentPush = pushIndex.load(std::memory_order_acquire);

            // 큐가 비어있는 상태 체크
            if (currentPop == currentPush) {
                if (stopSignal.load(std::memory_order_relaxed)) {
                    return false; // 동영상 종료 신호가 켜졌다면 탈출
                }
                std::this_thread::yield(); // 잠시 CPU를 양보하며 대기
                continue;
            }

            size_t index = currentPop % kCapacity;

            // cv::Mat일 때는 메모리 풀링, 일반 구조체는 대입 연산 사용
            if constexpr (std::is_same_v<T, cv::Mat>) {
                ringBuffer[index].copyTo(item);
            } else if constexpr (std::is_same_v<T, InferenceResult>) {
                // consumer 측 바구니(item.frame)에도 안전하게 픽셀 복사 전달
                ringBuffer[index].frame.copyTo(item.frame);
                item.detections = std::move(ringBuffer[index].detections);
            } else {
                item = std::move(ringBuffer[index]);
            }

            popIndex.store(currentPop + 1, std::memory_order_release);
            return true;
        }
    }
};

// 스레드 1: 동영상 전용 디코딩 스레드 (기존과 동일)
void videoDecodingThread(const std::string& videoPath, LockFreeQueue<cv::Mat>& lfQueueA, std::atomic<bool>& stopSignalA) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "비디오 파일을 열 수 없습니다: " << videoPath << std::endl;
        stopSignalA = true;
        return;
    }

    cv::Mat frame;
    while (!stopSignalA) {
        cap >> frame;
        if (frame.empty()) break; // 동영상 끝

        // 큐A가 가득 차서 push가 실패하면 자리가 생길 때까지 무한 대기
        while (!lfQueueA.push(std::move(frame), stopSignalA)) { // std::move 적용
            std::this_thread::yield();
        }
    }
    cap.release();
    stopSignalA = true; // 영상 처리가 끝났음을 큐A consumer에게 알림
}

// 스레드 2: GPU 추론 전용 스레드 함수 구현
void gpuInferenceThread(
    YoloTRT& detector, 
    LockFreeQueue<cv::Mat>& lfQueueA, 
    LockFreeQueue<InferenceResult>& lfQueueB, 
    std::atomic<bool>& stopSignalA, 
    std::atomic<bool>& stopSignalB
) {
    cv::Mat frame;
    // 추론 전용 프레임 메모리 버퍼 미리 파두기 (지역 변수 재할당 방지)
    frame = cv::Mat::zeros(640, 640, CV_8UC3);

    while (true) {
        // 큐 A에서 프레임을 꺼냄 (consumer)
        if (!lfQueueA.pop(frame, stopSignalA)) {
            break;
        }

        // 정확한 측정 시작점: GPU 추론 직전에 타이머 스타트
        // auto frameStart = std::chrono::high_resolution_clock::now();

        // 오직 GPU 추론만 수행 (enqueueV3)
        std::vector<Detection> detections = detector.detect(frame, 0.25f, 0.45f);

        // 결과 구조체 생성
        InferenceResult res;
        res.frame = frame;
        res.detections = std::move(detections);
        // res.startTime = frameStart; // 시간 토스

        // 큐B에 데이터 던짐(producer)
        // 핵심: push할 때 임시 우측값으로 변환(std::move)하여 큐B에 투하
        while (!lfQueueB.push(std::move(res), stopSignalB)) {
            std::this_thread::yield();
        }
    }
    // GPU 추론이 끝나면 NMS 스레드에게도 종료 신호를 전달
    stopSignalB = true;
}

// 스레드 3: 후처리(NMS) 및 시각화 전용 스레드 함수 구현
void cpuPostProcessingThread(
    LockFreeQueue<InferenceResult>& lfQueueB, 
    std::atomic<bool>& stopSignalB, 
    std::vector<double>& latencies, 
    int& frameCount
) {
    InferenceResult res;
    // consumer 측 바구니도 메모리 미리 확보하여 동적 할당 방지
    res.frame = cv::Mat::zeros(640, 640, CV_8UC3);

    while (true) {
        // 큐B에서 데이터를 꺼냄
        if (!lfQueueB.pop(res, stopSignalB)) {
            break;
        }

        for (const auto& det : res.detections) {
            cv::rectangle(res.frame, det.box, cv::Scalar(0, 255, 0), 2);
        }

        // auto frameEnd = std::chrono::high_resolution_clock::now();
        // double ms = std::chrono::duration<double, std::milli>(frameEnd - res.startTime).count();
        
        // latencies.push_back(ms);
        frameCount++;
    }
}

int main() {
    std::string engineName = "yolov8n.fp16.engine";

    YoloTRT detector(engineName);
    std::string videoPath = "./test_videos/sample.mp4";

    cv::VideoCapture tempCap(videoPath);
    int totalFrames = static_cast<int>(tempCap.get(cv::CAP_PROP_FRAME_COUNT));
    tempCap.release();

    cv::Mat dummyFrame = cv::Mat::zeros(640, 640, CV_8UC3);
    for (int i = 0; i < 50; i++) {
        detector.detect(dummyFrame, 0.25f, 0.45f);
    }

    LockFreeQueue<cv::Mat> lfQueueA; // 영상 읽기 -> GPU 추론
    LockFreeQueue<InferenceResult> lfQueueB; // GPU 추론 -> 후처리/시각화

    std::atomic<bool> stopSignalA{false};
    std::atomic<bool> stopSignalB{false};
    std::vector<double> latencies;
    // latencies.reserve(totalFrames);
    int frameCount = 0;

    auto totalStart = std::chrono::high_resolution_clock::now();

    // 스레드 3개 가동
    // 스레드 1: 디코딩
    std::thread reader(videoDecodingThread, videoPath, std::ref(lfQueueA), std::ref(stopSignalA));

    // 스레드 2: GPU 추론 전용 스레드 가동
    std::thread inferencer(gpuInferenceThread, std::ref(detector), std::ref(lfQueueA), std::ref(lfQueueB), std::ref(stopSignalA), std::ref(stopSignalB));

    // 스레드 3: CPU 후처리 및 시각화 스레드를 메인 스레드가 직접 수행하거나 별도 스레드로 분리
    // 여기서는 메인 스레드에서 직접 돌리거나 별도 스레드로 관리할 수 있습니다.
    std::thread transformer(cpuPostProcessingThread, std::ref(lfQueueB), std::ref(stopSignalB), std::ref(latencies), std::ref(frameCount));

    // 안전하게 스레드 합류
    if (reader.joinable()) reader.join();
    if (inferencer.joinable()) inferencer.join();
    if (transformer.joinable()) transformer.join();

    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalSec = std::chrono::duration<double>(totalEnd - totalStart).count();

    double realFPS = frameCount / totalSec;
    std::cout << "총 소요 시간: " << cv::format("%.2f", totalSec) << " 초" << std::endl;
    std::cout << "진짜 파이프라인 처리 속도: " << cv::format("%.1f", realFPS) << " FPS" << std::endl;

    // 최종 벤치마크 결과 계산 및 출력
    // double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    // double avg = sum / latencies.size();
    // double min = *std::min_element(latencies.begin(), latencies.end());
    // double max = *std::max_element(latencies.begin(), latencies.end());

    std::cout << "[Lock-Free 3 Threads 벤치마크]" << std::endl;
    std::cout << "가속 엔진: " << engineName << std::endl;
    std::cout << "처리 프레임 수: " << frameCount << " / " << totalFrames << std::endl;
    // std::cout << "평균 Latency (Avg): " << cv::format("%.2f", avg) << " ms" << std::endl;
    // std::cout << "최소 Latency (Min): " << cv::format("%.2f", min) << " ms" << std::endl;
    // std::cout << "최대 Latency (Max): " << cv::format("%.2f", max) << " ms" << std::endl;
    // std::cout << "최종 Lock-Free 3 Threads 처리 속도: " << cv::format("%.1f", 1000.0 / avg) << " FPS" << std::endl;

    return 0;
}