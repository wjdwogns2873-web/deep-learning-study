#include "YoloTRT.h"
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <memory>
#include <atomic>
// #include <random>
#include <opencv2/opencv.hpp>

// 각 프레임이 어느 카메라(스트림)에서 왔는지 식별하기 위한 패킷 구조체
struct FramePacket {
    int streamId; // 카메라 번호(0, 1, 2, ..)
    int frameNumber; // 해당 카메라의 프레임 순번
    std::shared_ptr<cv::Mat> imagePtr; // 제로카피를 위한 스마트 포인터
    std::vector<float> preprocessedData; // 카메라 스레드가 미리 전처리해둔 CHW
    cv::Size origSize;
};

// 여러 생산자(카메라)가 밀어넣은 프레임을 묶어주는 동적 배치 수집기
class DynamicBatchCollector {
private:
    std::vector<FramePacket> buffer_; // 프레임을 임시로 모아두는 바구니
    size_t maxBatchSize_; // 최대 배치 크기
    std::chrono::milliseconds timeout_; // 최대 대기 시간

    std::mutex mtx_; // 여러 카메라 스레드의 동시 접근을 막는 mutex
    std::condition_variable cvNotEmpty_; // 바구니에 데이터가 들어왔음을 추론 스레드에 알리는 신호
    std::condition_variable cvNotFull_; // 바구니가 비워졌음을 카메라 스레드에 알리는 신호
public:
    DynamicBatchCollector(size_t maxBatchSize, int timeoutMs)
        : maxBatchSize_(maxBatchSize), timeout_(std::chrono::milliseconds(timeoutMs)) {
            buffer_.reserve(maxBatchSize_); // 동적 할당 방지를 위해 미리 공간 예약
    }

    // producer: 여러 카메라 스레드가 동시에 호출하는 함수
    bool push(FramePacket&& packet, const std::atomic<bool>& stopSignal) {
        // unique_lock은 lock_guard와 달리 중간에 잠금을 풀거나 다시 잠글 수 있는 유연한 자물쇠입니다.
        std::unique_lock<std::mutex> lock(mtx_);

        // 바구니가 꽉 찼다면 자리가 생길 때까지 스레드를 재워서 CPU 낭비를 막습니다.
        cvNotFull_.wait(lock, [this, &stopSignal]() {
            return buffer_.size() < maxBatchSize_ || stopSignal.load();
        });

        if (stopSignal.load()) {
            return false;
        }

        // 소유권 이동함으로써 복사 없이 바구니에 적재
        buffer_.push_back(std::move(packet));

        // 바구니에 프레임이 들어왔으니 추론 스레드를 깨웁니다.
        cvNotEmpty_.notify_one();
        return true;
    }

    // consumer 호출: 추론 스레드가 배치 묶음을 통째로 꺼내가는 함수
    bool popBatch(std::vector<FramePacket>& outBatch, const std::atomic<bool>& stopSignal) {
        std::unique_lock<std::mutex> lock(mtx_);

        // 첫 번째 프레임이 들어올 때까지 대기합니다.
        cvNotEmpty_.wait(lock, [this, &stopSignal]() {
            return !buffer_.empty() || stopSignal.load();
        });

        if (buffer_.empty() && stopSignal.load()) {
            return false;
        }

        // 최대 배치 크기(4)가 모이거나, 제한 시간(10ms)이 지날 때까지만 기다립니다.
        cvNotEmpty_.wait_for(lock, timeout_, [this, &stopSignal]() {
            return buffer_.size() >= maxBatchSize_ || stopSignal.load();
        });

        // 바구니(buffer_)에 모인 프레임들의 소유권을 통째로 outBatch로 이동시킵니다.
        outBatch.swap(buffer_);

        // 바구니를 완전히 비웠으니 대기 중이던 카메라 스레드들을 모두 깨웁니다.
        cvNotFull_.notify_all();
        return true;
    }

    // 모든 카메라 종료 시 잠들어 있는 추론 스레드를 강제로 깨우는 함수
    void notifyStop() {
        std::lock_guard<std::mutex> lock(mtx_);
        cvNotEmpty_.notify_all();
        cvNotFull_.notify_all();
    }
};

// [producer] 실제 동영상 파일(가상 CCTV 스트림) 디코딩 스레드
void cameraStreamThread(
    int streamId, 
    const std::string& videoPath, 
    DynamicBatchCollector& collector, 
    const std::atomic<bool>& stopSignal
) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[카메라 " << streamId << "] 비디오 열기 실패\n";
        return;
    }

    int frameIdx = 0;
    while (!stopSignal.load()) {
        auto framePtr = std::make_shared<cv::Mat>();
        cap >> *framePtr;

        if (framePtr->empty()) break;

        frameIdx++;

        // std::move(framePtr)로 소유권을 넘기기 전에 원본 크기를 먼저 변수에 담아둡니다.
        cv::Size origSize = framePtr->size();

        // 각 카메라 스레드가 각자 병렬로 전처리 수행
        std::vector<float> prepBuffer(YoloTRT::kSingleInputElements);
        YoloTRT::preprocessSingle(*framePtr, prepBuffer);





        FramePacket packet{
            streamId, 
            frameIdx, 
            std::move(framePtr), 
            std::move(prepBuffer), 
            origSize
        };

        if (!collector.push(std::move(packet), stopSignal)) break;
    }
    cap.release();
}

// [consumer] TensorRT 동적 배치 추론 스레드
void batchInferenceThread(
    YoloTRT& detector, 
    DynamicBatchCollector& collector, 
    const std::atomic<bool>& stopSignal, 
    int& totalProcessedFrames, 
    int& totalBatches
) {
    std::vector<FramePacket> batch;
    // std::vector<cv::Mat> batchImages; // TensorRT에 넘길 이미지 묶음
    std::vector<std::vector<float>> batchBuffers;
    std::vector<cv::Size> origSizes;
    
    // 메모리 재할당 방지를 위해 미리 할당
    batch.reserve(4);
    batchBuffers.reserve(4);
    origSizes.reserve(4);

    while (true) {
        batch.clear();
        batchBuffers.clear();
        origSizes.clear();

        if (!collector.popBatch(batch, stopSignal)) {
            break;
        }

        // 패킷에서 전처리 완료 버퍼의 소유권만 이동
        for (auto& pkt : batch) {
            batchBuffers.push_back(std::move(pkt.preprocessedData));
            origSizes.push_back(pkt.origSize);
        }

        auto batchDetections = detector.detectBatchPreprocessed(batchBuffers, origSizes, 0.25f, 0.45f);

        for (size_t i = 0; i < batch.size(); ++i) {
            for (const auto& det : batchDetections[i]) {
                cv::rectangle(*batch[i].imagePtr, det.box, cv::Scalar(0, 255, 0), 2);
            }
        }

        totalProcessedFrames += static_cast<int>(batch.size());
        totalBatches++;
    }
}

int main() {
    std::string engineName = "yolov8n.dynamic.engine";
    YoloTRT detector(engineName);

    // 카메라 4대의 영상 경로 (테스트용으로 동일 영상을 4개 스트림으로 동시 입력)
    std::vector<std::string> videoPaths = {
        "./sample_data/sample.mp4", 
        "./sample_data/sample.mp4", 
        "./sample_data/sample.mp4", 
        "./sample_data/sample.mp4"
    };

    const size_t maxBatchSize = 4;
    const int timeoutMs = 15;

    std::vector<cv::Mat> dummyBatch(maxBatchSize, cv::Mat::zeros(640, 640, CV_8UC3));
    for (int i = 0; i < 30; ++i) {
        detector.detectBatch(dummyBatch, 0.25f, 0.45f);
    }

    DynamicBatchCollector collector(maxBatchSize, timeoutMs);
    std::atomic<bool> stopSignal{false};
    int totalProcessedFrames = 0;
    int totalBatches = 0;

    auto totalStart = std::chrono::high_resolution_clock::now();

    // 추론용 consumer 스레드 시작
    std::thread inferencer(
        batchInferenceThread, 
        std::ref(detector), 
        std::ref(collector), 
        std::ref(stopSignal), 
        std::ref(totalProcessedFrames), 
        std::ref(totalBatches)
    );

    // 카메라 4대 producer 스레드 시작
    std::vector<std::thread> cameraThreads;
    for (int i = 0; i < static_cast<int>(videoPaths.size()); ++i) {
        cameraThreads.emplace_back(
            cameraStreamThread, 
            i, 
            videoPaths[i], 
            std::ref(collector), 
            std::ref(stopSignal)
        );
    }

    // 모든 카메라 스트림이 끝날 때까지 대기
    for (auto& t : cameraThreads) {
        if (t.joinable()) t.join();
    }

    // 모든 카메라 종료 후 안전하게 추론 스레드 깨워서 종료
    stopSignal = true;
    collector.notifyStop();
    if (inferencer.joinable()) inferencer.join();

    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalSec = std::chrono::duration<double>(totalEnd - totalStart).count();

    std::cout << "총 처리 프레임 수: " << totalProcessedFrames << " 프레임" << std::endl;
    std::cout << "총 배치 실행 횟수: " << totalBatches << "회 (평균 배치 크기: "
              << cv::format("%.2f", static_cast<double>(totalProcessedFrames) / totalBatches) << "장)" << std::endl;
    std::cout << "총 소요 시간: " << cv::format("%.2f", totalSec) << " 초" << std::endl;
    std::cout << "전체 시스템 처리량: " << cv::format("%.1f", totalProcessedFrames / totalSec) << " FPS" << std::endl;

    return 0;
}