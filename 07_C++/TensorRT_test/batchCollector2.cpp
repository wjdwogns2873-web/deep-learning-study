#include "YoloTRT.h"
#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <memory>
#include <atomic>
#include <opencv2/opencv.hpp>

// Object Pool에서 재사용할 패킷 구조체
struct FramePacket {
    int streamId = 0;
    int frameNumber = 0;
    cv::Mat image;                       // 카메라 프레임이 덮어씌워질 버퍼
    std::vector<float> preprocessedData; // 전처리된 CHW 데이터가 덮어씌워질 버퍼 (4.91MB 고정)
    cv::Size origSize;

    // 생성 시점에 단 한 번만 4.91MB 메모리를 확보해 둡니다.
    FramePacket() : preprocessedData(YoloTRT::kSingleInputElements) {}
};

// 여러 카메라 스레드와 추론 스레드가 안전하게 패킷을 빌리고 반납하는 스레드 안전 메모리 풀
class PacketObjectPool {
private:
    std::queue<FramePacket*> pool_;
    std::vector<std::unique_ptr<FramePacket>> allPackets_; // 전체 패킷의 실제 수명 관리자
    std::mutex mtx_;
    std::condition_variable cvAvailable_;

public:
    // poolSize 개수만큼 패킷을 미리 생성해 풀에 채워 둡니다.
    explicit PacketObjectPool(size_t poolSize) {
        allPackets_.reserve(poolSize);
        for (size_t i = 0; i < poolSize; ++i) {
            auto pkt = std::make_unique<FramePacket>();
            pool_.push(pkt.get());
            allPackets_.push_back(std::move(pkt));
        }
    }

    // [생산자 호출] 풀에서 빈 패킷 하나를 빌려옵니다. (풀이 비어있으면 반납될 때까지 대기)
    std::shared_ptr<FramePacket> acquire(const std::atomic<bool>& stopSignal) {
        std::unique_lock<std::mutex> lock(mtx_);

        cvAvailable_.wait(lock, [this, &stopSignal]() {
            return !pool_.empty() || stopSignal.load();
        });

        if (pool_.empty() && stopSignal.load()) {
            return nullptr;
        }

        FramePacket* rawPtr = pool_.front();
        pool_.pop();

        // [핵심] 스마트 포인터의 참조 카운트가 0이 될 때 delete 대신 풀로 반납하는 커스텀 삭제자
        return std::shared_ptr<FramePacket>(rawPtr, [this](FramePacket* p) {
            std::lock_guard<std::mutex> returnLock(mtx_);
            pool_.push(p);
            cvAvailable_.notify_one(); // 반납되었음을 대기 중인 카메라 스레드에 알림
        });
    }

    void notifyStop() {
        std::lock_guard<std::mutex> lock(mtx_);
        cvAvailable_.notify_all();
    }
};

// 동적 배치 수집기 (무거운 데이터 이동 대신 가벼운 shared_ptr<FramePacket> 포인터만 이동)
class DynamicBatchCollector {
private:
    std::vector<std::shared_ptr<FramePacket>> buffer_;
    size_t maxBatchSize_;
    std::chrono::milliseconds timeout_;

    std::mutex mtx_;
    std::condition_variable cvNotEmpty_;
    std::condition_variable cvNotFull_;

public:
    DynamicBatchCollector(size_t maxBatchSize, int timeoutMs)
        : maxBatchSize_(maxBatchSize), timeout_(std::chrono::milliseconds(timeoutMs)) {
        buffer_.reserve(maxBatchSize_);
    }

    bool push(std::shared_ptr<FramePacket> packet, const std::atomic<bool>& stopSignal) {
        std::unique_lock<std::mutex> lock(mtx_);

        cvNotFull_.wait(lock, [this, &stopSignal]() {
            return buffer_.size() < maxBatchSize_ || stopSignal.load();
        });

        if (stopSignal.load()) {
            return false;
        }

        buffer_.push_back(std::move(packet));

        cvNotEmpty_.notify_one();
        return true;
    }

    bool popBatch(std::vector<std::shared_ptr<FramePacket>>& outBatch, const std::atomic<bool>& stopSignal) {
        std::unique_lock<std::mutex> lock(mtx_);

        cvNotEmpty_.wait(lock, [this, &stopSignal]() {
            return !buffer_.empty() || stopSignal.load();
        });

        if (buffer_.empty() && stopSignal.load()) {
            return false;
        }

        cvNotEmpty_.wait_for(lock, timeout_, [this, &stopSignal]() {
            return buffer_.size() >= maxBatchSize_ || stopSignal.load();
        });

        outBatch.swap(buffer_);

        cvNotFull_.notify_all();
        return true;
    }

    void notifyStop() {
        std::lock_guard<std::mutex> lock(mtx_);
        cvNotEmpty_.notify_all();
        cvNotFull_.notify_all();
    }
};

// [producer] Object Pool에서 패킷을 빌려 재사용하는 카메라 스레드
void cameraStreamThread(
    int streamId, 
    const std::string& videoPath, 
    PacketObjectPool& pool,
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
        // 1. new/make_shared 없이 풀에서 이미 할당된 패킷을 빌려옵니다.
        auto packet = pool.acquire(stopSignal);
        if (!packet) break;

        // 2. 기존 packet->image 버퍼 위에 새 프레임을 그대로 덮어씁니다. (메모리 재할당 없음)
        cap >> packet->image;
        if (packet->image.empty()) break;

        frameIdx++;
        packet->streamId = streamId;
        packet->frameNumber = frameIdx;
        packet->origSize = packet->image.size();

        // 3. 이미 4.91MB가 확보된 packet->preprocessedData 위에 전처리 결과만 덮어씁니다.
        YoloTRT::preprocessSingle(packet->image, packet->preprocessedData);

        if (!collector.push(std::move(packet), stopSignal)) break;
    }
    cap.release();
}

// [consumer] 추론 후 패킷을 자동으로 풀에 반납하는 추론 스레드
void batchInferenceThread(
    YoloTRT& detector, 
    DynamicBatchCollector& collector, 
    const std::atomic<bool>& stopSignal, 
    int& totalProcessedFrames, 
    int& totalBatches
) {
    std::vector<std::shared_ptr<FramePacket>> batch;
    std::vector<std::vector<float>> batchBuffers;
    std::vector<cv::Size> origSizes;

    batch.reserve(4);
    batchBuffers.reserve(4);
    origSizes.reserve(4);

    while (true) {
        // [중요] 이전 배치의 batch.clear()가 실행되는 순간,
        // 각 shared_ptr의 커스텀 삭제자가 호출되어 패킷들이 자동으로 Object Pool에 반납됩니다!
        batch.clear();
        batchBuffers.clear();
        origSizes.clear();

        if (!collector.popBatch(batch, stopSignal)) {
            break;
        }

        // 주의: 여기서는 std::move(pkt->preprocessedData)를 절대 쓰면 안 됩니다!
        // move를 하면 풀에 반납될 버퍼의 용량(capacity)이 0으로 뺏겨버리기 때문입니다.
        // 대신 포인터/참조만 전달하거나 복사 없이 읽을 수 있도록 구성합니다.
        for (const auto& pkt : batch) {
            batchBuffers.push_back(pkt->preprocessedData);
            origSizes.push_back(pkt->origSize);
        }

        auto batchDetections = detector.detectBatchPreprocessed(batchBuffers, origSizes, 0.25f, 0.45f);

        for (size_t i = 0; i < batch.size(); ++i) {
            for (const auto& det : batchDetections[i]) {
                cv::rectangle(batch[i]->image, det.box, cv::Scalar(0, 255, 0), 2);
            }
        }

        totalProcessedFrames += static_cast<int>(batch.size());
        totalBatches++;
    }
}

int main() {
    std::string engineName = "yolov8n.dynamic.engine";
    YoloTRT detector(engineName);

    std::vector<std::string> videoPaths = {
        "./sample_data/sample.mp4", 
        "./sample_data/sample.mp4", 
        "./sample_data/sample.mp4", 
        "./sample_data/sample.mp4"
    };

    const size_t maxBatchSize = 4;
    const int timeoutMs = 15;
    // 카메라 4대가 동시에 작업할 공간(4개) + 바구니 대기(4개) + 추론 중(4개) + 여유분 = 16개
    const size_t poolSize = 16;

    std::vector<cv::Mat> dummyBatch(maxBatchSize, cv::Mat::zeros(640, 640, CV_8UC3));
    for (int i = 0; i < 30; ++i) {
        detector.detectBatch(dummyBatch, 0.25f, 0.45f);
    }

    PacketObjectPool pool(poolSize);
    DynamicBatchCollector collector(maxBatchSize, timeoutMs);
    std::atomic<bool> stopSignal{false};
    int totalProcessedFrames = 0;
    int totalBatches = 0;

    auto totalStart = std::chrono::high_resolution_clock::now();

    std::thread inferencer(
        batchInferenceThread, 
        std::ref(detector), 
        std::ref(collector), 
        std::ref(stopSignal), 
        std::ref(totalProcessedFrames), 
        std::ref(totalBatches)
    );

    std::vector<std::thread> cameraThreads;
    for (int i = 0; i < static_cast<int>(videoPaths.size()); ++i) {
        cameraThreads.emplace_back(
            cameraStreamThread, 
            i, 
            videoPaths[i], 
            std::ref(pool),
            std::ref(collector), 
            std::ref(stopSignal)
        );
    }

    for (auto& t : cameraThreads) {
        if (t.joinable()) t.join();
    }

    stopSignal = true;
    pool.notifyStop();
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
