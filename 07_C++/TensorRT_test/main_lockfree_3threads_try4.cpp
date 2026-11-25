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

// 네 번째 시도(Try4)
// std::make_shared에 숨겨진 동적 할당과 lock 병목
// 스마트 포인터 자체는 가볍지만, 매 루프마다 std::make_shared를 호출하면, CPU는 OS에게 새로운 heap 메모리 공간을 달라고 요청해야 합니다.
// C++ 런타임의 힙 메모리 할당자는 내부적으로 멀티스레드 안전을 위해 mutex나 lock을 잡고 동작합니다.
// 즉, 우리는 lock-free 코드를 짰다고 생각했지만, make_shared를 매 프레임 호출하는 순간 OS 레벨에서 매번 lock이 걸려
// 스레드가 대기하게 되면서 lock-free의 이점이 사라진 것입니다.

// 해결 전략: Lock free 기반 Object Pool 도입
// Try3의 매번 make_shared하는 비용도 줄이기 위해 32개의 포인터 방을 미리 파두고, 그걸 Object Pool이라고 하는데, 
// 하나씩 꺼내서(pop) 나온 주소를 가리키게 하고, 추론 및 후처리가 끝나면 재사용하기 위해 다시 Object Pool에 넣어두는 전략입니다.

// 그럼에도 Try3보다 근소하게 느려진 이유를 분석해 보았습니다.
// Object Pool을 도입하면서 Lock-Free 큐가 총 3개로 늘어났고, 
// 매 프레임마다 원자적 연산(atomic, load/store) 등으로 인해 make_shared 제거로 얻은 이득이 사라졌습니다.
// 다만 메모리 할당량이 32개로 고정되어 있으므로 장시간 실행시 메모리 단편화를 막는 안정성 면에서는 안정성이 놆습니다.

// 메모리 단편화(Memory Fragmentation)란?
// 전체 빈 메모리 공간은 충분하지만, 빈 공간들이 자잘한 조각으로 쪼개져 있어서 정작 큰 데이터를 한번에 넣지 못하는 현상입니다.

// 주차장 비유로 설명하자면,
// 총 10대의 차를 댈 수 있는 주차장이 있습니다. 현재 5대가 주차되어 있고 빈자리는 5칸입니다.
// 만약 빈자리가 [차] [빈칸] [차] [빈칸] [차] [빈칸] [차] [빈칸] [차] [빈칸] 처럼 뛰엄뛰엄 되어있다면, 
// 연속된 3칸이 필요한 리무진(큰 데이터)은 주차할 수 없습니다. 컴퓨터의 힙 메모리에서도 똑같은 일이 벌어집니다.

// 짧은 동영상을 테스트할 때는 티가 안 나지만, 실제 서비스 환경에서는 큰 문제가 생길 수 있습니다.
// 1. 간헐적인 렉 발생: 메모리 단편화가 심해지면 메모리 할당자가 빈자리를 찾거나 조각난 빈칸들을 합치느라 시간을 씁니다.
// 이로 인해 추론 시간이 갑자기 튀는 현상이 발생합니다.
// 2. 장시간 실행시 프로그램 강제 종료: 공장이나 CCTV처럼 24시간 내내 돌아가는 시스템에서 전체 남은 RAM이 2GB나 되는데도
// 연속된 1.2MB 공간이 없어서 메모리 부족 에러로 프로그램이 종료될 수 있습니다.

// 추론 결과 및 추론 측정 시간을 담는 구조체입니다.
// frame을 스마트 포인터로 변경합니다.
struct InferenceResult {
    std::shared_ptr<cv::Mat> frame; // 추론할 때 넣은 프레임입니다. 이 프레임 위에 rectangle을 하기 위해 존재합니다.
    std::vector<Detection> detections; // 탐지 결과를 담는 detections입니다.
    std::chrono::high_resolution_clock::time_point startTime; // 추론 직전 시간을 담기 위해 존재합니다.
};

// cv::Mat 배열, InferenceResult 배열을 만들기 위해 템플릿을 선언합니다.
template <typename T>
class LockFreeQueue {
private:
    static constexpr size_t kCapacity = 32; // 링 버퍼의 최대 용량을 설정합니다.
    T ringBuffer[kCapacity]; // 이 배열은 Mat 객체가 아니라 8바이트 포인터(또는 가벼운 구조체)만 담습니다.

    // mutex를 쓰지 않고, 원자적 변수를 선언합니다.
    // pushIndex는 push할 때, popIndex는 pop할 때 사용합니다.
    std::atomic<size_t> pushIndex{0};
    std::atomic<size_t> popIndex{0};

public:
    // 생성자입니다.
    LockFreeQueue() {
        // 포인터나 move 전용 구조체를 담으므로, 생성자에서 zeros로 방을 미리 파둘 필요가 없습니다.
    }

    // producer: 큐에 프레임을 넣습니다(push)
    // 넣을 수 있는 상황(큐의 공간이 남아있음)이면 true, 넣을 수 없는 상황(큐가 가득 찼음)이면 false를 반환합니다.
    // std::move를 통해 우측값 참조로 소유권만 넘겨받아 픽셀 복사를 원천 차단합니다.
    bool push(T&& item, std::atomic<bool>& stopSignal) {
        // 각각 현재 큐에 넣을(push) 방의 번호, 큐에서 꺼내갈(pop) 방의 번호를 의미합니다. 이 번호는 32를 넘어 동영상 프레임의 개수까지 증가합니다.
        size_t currentPush = pushIndex.load(std::memory_order_relaxed);
        size_t currentPop = popIndex.load(std::memory_order_acquire);

        // 큐가 가득 찼는지 체크합니다.
        // 녛을(push) 프레임(또는 구조체)의 번호가 67이고 꺼내갈(pop) 프레임의 번호가 35라면 큐에는 32(67-35)개의 데이터가 있기에 가득 차 있는 상태입니다.
        // 하지만 녛을(push) 프레임의 번호가 67이고 꺼내갈(pop) 프레임의 번호가 34라면 큐에는 31(67-34)개의 데이터가 있기에 1개를 더 넣을 수 있는 상태가 되고, 
        // if문을 타지 않고 밑으로 내려갑니다. 넣을(push) 준비를 하는 것입니다.
        if ((currentPush - currentPop) >= kCapacity) {
            return false;
        }

        // 방 32개를 순환해서 쓰고 있기 때문에, 현재 넣을(push) 프레임 번호를 kCapacity(32)로 나눈 나머지가 index가 됩니다.
        size_t index = currentPush % kCapacity;

        // 무거운 copyTo를 전부 걷어내고, 오직 포인터 주소/소유권만 이동(std::move)시킵니다.
        ringBuffer[index] = std::move(item);

        // 이제 다음 넣을 프레임의 번호는 67 + 1 = 68이 됩니다.
        pushIndex.store(currentPush + 1, std::memory_order_release);
        // 넣기(push)를 완료했으니 true를 반환합니다.
        return true;
    }

    // consumer: 큐에서 데이터 추출
    // 링버퍼에서 꺼낸 것을 item에 담습니다.
    // pop은 push와 다른 점이, while문으로 이루어져 있습니다.
    // 따라서 pop 함수 하나만 켜 두어도 while문이 계속 돌아가는 것입니다.
    // 
    bool pop(T& item, std::atomic<bool>& stopSignal) {
        while (true) {
            size_t currentPop = popIndex.load(std::memory_order_relaxed);
            size_t currentPush = pushIndex.load(std::memory_order_acquire);

            // 큐가 비어있는지 체크합니다.
            // 꺼낼(pop) 프레임(또는 구조체)의 번호가 넣을(push) 프레임의 번호와 같다면 큐가 비어있다는 뜻입니다.
            if (currentPop == currentPush) {
                if (stopSignal.load(std::memory_order_relaxed)) {
                    return false; // stopSignal이 true라면 종료 신호가 켜졌다는 걸 의미하므로, while 문을 탈출합니다.
                }
                std::this_thread::yield(); // 비어있지만 아직 종료 신호가 켜지지 않은 상태라면 CPU를 양보하며 대기합니다. 큐에 데이터가 들어올 때까지
                continue;
            }
            // 방 32개를 순환해서 쓰고 있기 때문에, 현재 꺼낼(pop) 프레임 번호를 kCapacity(32)로 나눈 나머지가 index가 됩니다.
            size_t index = currentPop % kCapacity;

            // consumer가 데이터를 가져갈 때도 오직 주소 소유권만 쏙 빼내어 넘겨줍니다.
            item = std::move(ringBuffer[index]);

            // 꺼내기(pop)를 완료했으므로, 더하기 1을 해줍니다.
            // 지금 67번 째를 꺼냈다면, 이제 다음 타겟은 68번 째 프레임(또는 구조체)입니다.
            popIndex.store(currentPop + 1, std::memory_order_release);
            return true;
        }
    }
};

// 스레드 1: 동영상 전용 디코딩 스레드
void videoDecodingThread(
    const std::string& videoPath, 
    LockFreeQueue<std::shared_ptr<cv::Mat>>& lfQueueA, 
    LockFreeQueue<std::shared_ptr<cv::Mat>>& objectPool, 
    std::atomic<bool>& stopSignalA) {
    // 비디오 파일 경로를 통해 VideoCapture 객체를 생성합니다.
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "비디오 파일을 열 수 없습니다: " << videoPath << std::endl;
        stopSignalA = true;
        return;
    }

    std::shared_ptr<cv::Mat> framePtr;

    // 종료 신호(stopSignalA)가 false(! 붙이면 true)일 때만 VideoCapture 객체에서 프레임을 꺼내고 꺼낸 프레임을 큐에 집어넣는(push) 작업을 진행합니다.
    while (!stopSignalA) {
        // make_shared를 제거. 오브젝트 풀 큐에서 놀고 있는 방 주소를 꺼내옵니다.
        while (!objectPool.pop(framePtr, stopSignalA)) {
            if (stopSignalA.load()) return;
            std::this_thread::yield();
        }

        cap >> *framePtr; // 꺼내온 기존 메모리 공간에 픽셀값만 그대로 덮어쓰기

        if (framePtr->empty()) {
            // 빈 프레임이면 사용하려던 포인터를 다시 풀에 돌려주고 탈출
            objectPool.push(std::move(framePtr), stopSignalA);
            break;
        }

        while (!lfQueueA.push(std::move(framePtr), stopSignalA)) {
            // 여기서 continue를 하면 안되는가?
            // continue를 하게 되면 중간에 프레임을 건너뛰게 되어 프레임 드랍이 발생할 수 있다.
            // 그래서 절대 건너뛰면 안되고, 빈자리가 생길 때까지 CPU를 양보(yield)하고, 대기한다.
            std::this_thread::yield();
        }
    }
    cap.release();
    // 종료 신호를 켭니다.
    stopSignalA = true;
}

// 스레드 2: 오직 GPU 추론만 전담하는 스레드
void gpuInferenceThread(
    YoloTRT& detector, 
    LockFreeQueue<std::shared_ptr<cv::Mat>>& lfQueueA, 
    LockFreeQueue<InferenceResult>& lfQueueB, 
    std::atomic<bool>& stopSignalA, 
    std::atomic<bool>& stopSignalB) {
    
    std::shared_ptr<cv::Mat> framePtr;
    while (true) {
        // 큐A에서 프레임을 꺼냄 (consumer)
        // pop()이 false라는 것은 stopSignalA(종료신호)가 켜졌음을 의미하므로 이제 이 while문을 중단시켜야 합니다.

        // 큐A에서 이미지 주소(shared_ptr)만 쏙 빼옵니다.
        if (!lfQueueA.pop(framePtr, stopSignalA)) {
            break;
        }
        // 추론 시작 직전에 시간 측정을 시작합니다.
        auto frameStart = std::chrono::high_resolution_clock::now();

        // 역참조(*framePtr)를 통해 실메모리에 접근하여 GPU 추론 진행
        std::vector<Detection> detections = detector.detect(*framePtr, 0.25f, 0.45f);

        // 추론 결과를 구조체에 담습니다.
        InferenceResult res;
        res.frame = std::move(framePtr); // 주소 소유권을 구조체로 이동
        res.detections = std::move(detections);
        res.startTime = frameStart;

        // 큐B에 추론 결과(구조체)를 넣습니다(push). (producer)
        // 이것 역시 큐의 빈자리가 나올 때까지 무한 대기합니다.
        while (!lfQueueB.push(std::move(res), stopSignalB)) {
            std::this_thread::yield();
        }
    }
    // 위의 pop() 함수가 false가 되었다 -> stopSignalA가 true가 되었다 -> 더이상 꺼낼 게 없으니 추론을 진행할 것도 없다
    // -> stopSignalB도 true로 전환.
    stopSignalB = true;
}

// 스레드 3: NMS, 시각화 및 Latency 측정
void cpuPostProcessingThread(
    LockFreeQueue<InferenceResult>& lfQueueB, 
    LockFreeQueue<std::shared_ptr<cv::Mat>>& objectPool, 
    std::atomic<bool>& stopSignalB, 
    std::vector<double>& latencies, 
    int& frameCount) {

    InferenceResult res;
    while (true) {
        // 큐B에서 데이터를 꺼냅니다. (consumer)
        // 큐B에서 결과 구조체를 통째로 소유권만 빼옵니다. 카피 부하 0
        if (!lfQueueB.pop(res, stopSignalB)) {
            break;
        }

        // 시각화(rectangle)를 진행합니다.
        // 받아온 주소 공간 상의 이미지에 직접 사각형을 그립니다.
        for (const auto& det : res.detections) {
            cv::rectangle(*res.frame, det.box, cv::Scalar(0, 255, 0), 2);
        }
        // 후처리 과정도 끝이 났으므로 시간 측정을 종료합니다.
        auto frameEnd = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(frameEnd - res.startTime).count();

        latencies.push_back(ms);
        frameCount++;

        // 처리가 끝난 이미지 포인터 주소를 파괴(free)하지 않고, 오브젝트 풀에 다시 반납(push)합니다.
        // 이로써 무한히 순환하는 Zero-Allocation 컨베이어 벨트가 완성됩니다.
        std::shared_ptr<cv::Mat> recycleMat = std::move(res.frame);
        while (!objectPool.push(std::move(recycleMat), stopSignalB)) {
            std::this_thread::yield();
        }
    }
}

int main() {
    std::string engineName = "yolov8n.int8.engine"; // 실행할 엔진을 선택합니다.
    // std::string engineName = "yolov8n.fp16.engine";
    // std::string engineName = "yolov8n.int8.engine";
    YoloTRT detector(engineName);

    std::string videoPath = "./test_videos/sample.mp4"; // 비디오 경로입니다.
    cv::VideoCapture tempCap(videoPath);
    int totalFrames = static_cast<int>(tempCap.get(cv::CAP_PROP_FRAME_COUNT));
    tempCap.release();

    // Warm-Up
    cv::Mat dummyFrame = cv::Mat::zeros(640, 640, CV_8UC3);
    for (int i = 0; i < 50; i++) {
        detector.detect(dummyFrame, 0.25f, 0.45f);
    }

    LockFreeQueue<std::shared_ptr<cv::Mat>> lfQueueA; // 큐A. VideoCapture에서 나온 프레임 하나하나를 담을 큐입니다.
    LockFreeQueue<InferenceResult> lfQueueB; // 큐B. 추론이 완료된 추론 결과(구조체)를 담을 큐입니다.


    // 오브젝트 풀 큐 생성: 놀고 있는 빈 포인터 방 32개를 담아둘 큐
    LockFreeQueue<std::shared_ptr<cv::Mat>> objectPool;
    std::atomic<bool> dummySignal{false};

    // 32개의 Mat 공간을 미리 힙에 할당하여 오브젝트 풀에 채워 넣습니다. (동적 할당 전면 배제)
    for (int i = 0; i < 32; i++) {
        auto allocatedMat = std::make_shared<cv::Mat>(640, 640, CV_8UC3);
        objectPool.push(std::move(allocatedMat), dummySignal);
    }


    std::atomic<bool> stopSignalA{false}; // 처음엔 false이지만, 프레임을 다 꺼냈다면 true가 됩니다.
    std::atomic<bool> stopSignalB{false}; // 처음엔 false이지만, 추론 결과(구조체)를 다 꺼내면 true가 됩니다.

    std::vector<double> latencies; // 각 프레임별 처리 시간을 담을 벡터입니다.
    latencies.reserve(totalFrames); // 불필요한 메모리 동적 할당을 막기 위해 총 프레임 개수 만큼 미리 메모리 공간을 확보합니다.
    int frameCount = 0;

    // 스레드 3개를 동시에 시작합니다.
    // reader: 비디오 파일에서 프레임을 하나씩 꺼내서 큐A에 넣는(push) 스레드
    // inferencer: 큐A에서 프레임을 꺼내서(pop) 추론을 진행하고, 추론 결과(구조체)를 큐B에 넣는(push) 스레드
    // postPorcessor: 큐B에서 추론 결과(구조체)를 꺼내서(pop) 후처리(시각화)를 하는 스레드
    std::thread reader(videoDecodingThread, videoPath, std::ref(lfQueueA), std::ref(objectPool), std::ref(stopSignalA));
    std::thread inferencer(gpuInferenceThread, std::ref(detector), std::ref(lfQueueA), std::ref(lfQueueB), std::ref(stopSignalA), std::ref(stopSignalB));
    std::thread postProcessor(cpuPostProcessingThread, std::ref(lfQueueB), std::ref(objectPool), std::ref(stopSignalB), std::ref(latencies), std::ref(frameCount));

    // 모든 스레드가 안전하게 종료될 때까지 대기합니다.
    if (reader.joinable()) reader.join();
    if (inferencer.joinable()) inferencer.join();
    if (postProcessor.joinable()) postProcessor.join();

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();
    double min = *std::min_element(latencies.begin(), latencies.end());
    double max = *std::max_element(latencies.begin(), latencies.end());

    std::cout << "사용한 가속 엔진: " << engineName << std::endl;
    std::cout << "총 처리 프레임 수: " << frameCount << std::endl;
    std::cout << "평균 Latency: " << avg << " ms" << std::endl;
    std::cout << "최소 Latency: " << min << " ms" << std::endl;
    std::cout << "최대 Latency: " << max << " ms" << std::endl;
    std::cout << "FPS: " << cv::format("%.1f", 1000.0 / avg) << " FPS" << std::endl;

    return 0;
}