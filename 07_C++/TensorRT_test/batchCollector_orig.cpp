#include "YoloTRT_orig.h"
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <memory>
#include <atomic>
#include <opencv2/opencv.hpp>

// 실제 이미지(cv::Mat)의 포인터가 담긴 구조체입니다.
struct FramePacket {
    int streamId; // 카메라 ID
    int frameNumber; // 프레임 번호(ex: 1 ~ 3129)
    std::shared_ptr<cv::Mat> imagePtr; // 제로카피를 위한 이미지의 포인터
};

// 배치를 모으고(push), 추론하는(popBatch) 클래스입니다.
class DynamicBatchCollector {
private:
    std::vector<FramePacket> buffer_; // 프레임 패킷을 배치 단위로 모을 벡터입니다.
    size_t maxBatchSize_; // 배치의 최대 개수입니다. 여기서는 4로 했습니다.
    std::chrono::milliseconds timeout_; // 한 배치에 4개가 쌓이지 않아도 일정 시간이 지나면 태워서 추론에 보내도록 타임아웃 시간을 설정합니다.
    std::mutex mtx_; // 여러 스레드의 동시 접근을 막는 mutex입니다.

    // 버퍼(buffer_)에 데이터가 들어올 때까지 추론 스레드를 재우고 깨우는 조건변수입니다.
    std::condition_variable cvNotEmpty_;
    // 버퍼에 빈 공간이 생길 때까지 카메라 스레드를 재우는 조건변수입니다.
    std::condition_variable cvNotFull_;
public:
    // 생성자: 멤버 초기화 리스트(:)를 사용해 변수들을 초기화합니다.
    // 불필요한 기본 생성 후 대입 과정을 생략해서 성능을 높입니다.
    DynamicBatchCollector(size_t maxBatchSize, int timeoutMs)
        : maxBatchSize_(maxBatchSize), timeout_(std::chrono::milliseconds(timeoutMs)) {
        buffer_.reserve(maxBatchSize_); // 동적 할당 방지를 위해 메모리 공간을 미리 예약합니다.
    }

    // 프레임 패킷을 우측값 참조(&&)로 받습니다. 소유권 이동(std::move)를 하기 위해서입니다.
    bool push(FramePacket&& packet, const std::atomic<bool>& stopSignal) {
        std::unique_lock<std::mutex> lock(mtx_); // 다른 스레드가 접근하지 못하게 자물쇠를 걸어 잠급니다.

        // 버퍼(buffer_)가 가득차지 않았다면(공간이 있다면) 밑으로 통과하게 합니다.
        // 공간이 없다면, 공간이 생길 때까지 스레드를 재워서 CPU 낭비를 막습니다.
        cvNotFull_.wait(lock, [this, &stopSignal]() {
            return buffer_.size() < maxBatchSize_ || stopSignal.load();
        });

        if (stopSignal.load()) return false;

        buffer_.push_back(std::move(packet)); // 소유권 이동함으로써, 복사비용 없이 버퍼(buffer_)에 적재합니다.
        cvNotEmpty_.notify_one(); // 버퍼에 프레임 패킷이 들어왔으니 추론 스레드를 깨웁니다.
        return true;
    }

    // 바꿔치기(swap)하기 위해 빈 벡터를 가져옵니다.
    bool popBatch(std::vector<FramePacket>& outBatch, const std::atomic<bool>& stopSignal) {
        std::unique_lock<std::mutex> lock(mtx_); // 다른 스레드가 접근하지 못하게 자물쇠를 걸어 잠급니다.

        // 버퍼(buffer_)가 비어있지 않다면(프레임이 존재한다면) 밑으로 통과하게 합니다.
        // 프레임이 존재하지 않는다면 생길 때까지 스레드를 재워서 CPU 낭비를 막습니다.
        cvNotEmpty_.wait(lock, [this, &stopSignal]() {
            return !buffer_.empty() || stopSignal.load();
        });

        // 버퍼(buffer_)가 비어있고, 종료신호가 켜졌다면 false를 반환합니다.
        if (buffer_.empty() && stopSignal.load()) return false;

        // 버퍼(buffer_) 안에 4개 이상(4개를 초과할 수 없기 때문에 사실상 4개가 맥시멈)이거나 지정한 시간(timeout_)이 지났을 때 밑으로 통과하게 합니다.
        cvNotEmpty_.wait_for(lock, timeout_, [this, &stopSignal]() {
            return buffer_.size() >= maxBatchSize_ || stopSignal.load();
        });

        // 버퍼(buffer_)에 모인 프레임들의 소유권을 통째로 outBatch로 이동(swap)시킵니다.
        // outBatch는 빈 벡터였으므로, 바꿔치기 함으로써 buffer_는 빈 벡터가 됩니다.
        outBatch.swap(buffer_);

        // 버퍼를 완전히 비웠으니 대기 중이던 카메라 스레드들을 모두 깨웁니다.
        cvNotFull_.notify_all();
        return true;
    }

    void notifyStop() {
        std::lock_guard<std::mutex> lock(mtx_);
        cvNotEmpty_.notify_all();
        cvNotFull_.notify_all();
    }
};

// 카메라 4대가 쓰는 스레드입니다. VideoCapture 객체에서 나온 프레임을 프레임 패킷 구조체에 담아서 push합니다.
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
        auto framePtr = std::make_shared<cv::Mat>(); // 스마트 포인터 그릇을 만듭니다.
        cap >> *framePtr; // 역참조하여서 VideoCapture에서 나온 프레임을 받습니다. 알맹이에 접근합니다.

        if (framePtr->empty()) break;

        frameIdx++;
        // shared_ptr의 참조 카운트 증가 비용조차 아끼기 위해 소유권을 이동합니다.
        FramePacket packet{ streamId, frameIdx, std::move(framePtr) };

        // push 함수에서 packet을 우측값 참조(&&)로 받고있기 때문에, std::move()로 넘겨줍니다.
        if (!collector.push(std::move(packet), stopSignal)) break;
    }
    cap.release();
}

// 배치 단위로 들어온 프레임을 consumer가 추론하는 스레드입니다.
void batchInferenceThread(
    YoloTRT& detector, 
    DynamicBatchCollector& collector, 
    const std::atomic<bool>& stopSignal, 
    int& totalProcessedFrames, 
    int& totalBatches, 
    const size_t maxBatchSize
) {
    std::vector<FramePacket> batch; // 바꿔치기(swap)하기 위해 벡터 껍데기를 준비해둡니다.
    std::vector<cv::Mat> batchImages; // 추론시 cv::Mat만 배치로 묶어서 보내기 위해 벡터를 준비합니다.
    batch.reserve(maxBatchSize); // 동적 할당을 막기 위해 미리 공간을 예약합니다.
    batchImages.reserve(maxBatchSize);

    while (true) {
        // 새로운 배치를 받기 전에 항상 비워줍니다.
        batch.clear();
        batchImages.clear();

        if (!collector.popBatch(batch, stopSignal)) break; // 바꿔치기(swap) 되었습니다.

        // cv::Mat의 스마트 포인터를 역참조(*)하여 알맹이만 batchImages 벡터에 담습니다.
        // cv::Mat은 대입연산자를 쓰거나 push_back을 할 때 데이터 통째로 복사하는 깊은복사가 아닌, 
        // 주소만 공유하는 얕은 복사가 됩니다. 그렇기 때문에 std::move를 쓰지 않았더라도 복사비용이 크지 않습니다.
        for (const auto& pkt : batch) {
            batchImages.push_back(*pkt.imagePtr);
        }

        // 담은 알맹이는 배치단위로 추론하는 데에 쓰입니다.
        // 실제 여기서의 자료형은 std::vector<std::vector<Detection>>이지만, 길기 때문에 auto로 했습니다.
        auto batchDetections = detector.detectBatch(batchImages, 0.25f, 0.45f);

        // 배치 개수 만큼 바운딩박스 그리기 작업을 해줍니다.
        // 여기 역시 역참조(*)하여 알맹이를 가져옵니다. cv::rectangle 함수에서는 알맹이를 필요로 합니다.
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
    std::string engineName = "yolov8n.dynamic_b4.int8.engine";
    YoloTRT detector(engineName);
    std::cout << "사용한 엔진: " << engineName << std::endl;

    // 테스트용으로 같은 비디오를 4개 사용합니다. 카메라도 4대라고 가정하는 겁니다.
    std::vector<std::string> videoPaths = {
        "./sample_data/sample.mp4", 
        "./sample_data/sample.mp4", 
        "./sample_data/sample.mp4", 
        "./sample_data/sample.mp4"
    };

    const size_t maxBatchSize = 4;
    const int timeoutMs = 15; // 15ms 동안 배치에 4개가 채워지지 않으면 지체하지 않고 태워서 추론으로 보냅니다.

    // Warm up 합니다.
    std::vector<cv::Mat> dummyBatch(maxBatchSize, cv::Mat::zeros(640, 640, CV_8UC3));
    for (int i = 0; i < 30; ++i) {
        detector.detectBatch(dummyBatch, 0.25f, 0.45f);
    }

    DynamicBatchCollector collector(maxBatchSize, timeoutMs);
    std::atomic<bool> stopSignal{false};
    int totalProcessedFrames = 0;
    int totalBatches = 0;

    auto totalStart = std::chrono::high_resolution_clock::now();

    // 추론 스레드 시작합니다. consumer
    std::thread inferencer(
        batchInferenceThread, 
        std::ref(detector), 
        std::ref(collector), 
        std::ref(stopSignal), 
        std::ref(totalProcessedFrames), 
        std::ref(totalBatches), 
        maxBatchSize
    );

    // 총 4개의 카메라 스레드를 시작합니다. producer
    std::vector<std::thread> cameraThreads;
    for (int i = 0; i < static_cast<int>(videoPaths.size()); ++i) {
        cameraThreads.emplace_back(cameraStreamThread, i, videoPaths[i], std::ref(collector), std::ref(stopSignal));
    }

    // 모든 카메라 스트림이 끝날 때까지 대기합니다.
    for (auto& t : cameraThreads) {
        if (t.joinable()) t.join();
    }

    // 모든 카메라 종료 후 추론 스레드를 깨워서 종료시킵니다.
    stopSignal = true;
    collector.notifyStop();
    if (inferencer.joinable()) inferencer.join();

    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalSec = std::chrono::duration<double>(totalEnd - totalStart).count();

    std::cout << "총 처리 프레임 수: " << totalProcessedFrames << " 프레임" << std::endl;
    std::cout << "총 배치 횟수: " << totalBatches << "회 (평균 배치 크기: "
              << cv::format("%.2f", static_cast<double>(totalProcessedFrames) / totalBatches) << "장)" << std::endl;
    std::cout << "총 소요 시간: " << cv::format("%.2f", totalSec) << "초" << std::endl;
    std::cout << "평균 FPS: " << cv::format("%.1f", totalProcessedFrames / totalSec) << " FPS" << std::endl;
}