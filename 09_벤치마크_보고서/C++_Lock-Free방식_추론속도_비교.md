# C++ TensorRT: FP32 vs FP16 vs INT8 동영상 스트리밍 벤치마크 보고서 (lock-free 방식)


## 📊 종합 성능 비교 요약

| 정밀도 엔진 | 평균 Latency (Avg) | 최소 Latency (Min) | 최대 Latency (Max) | 평균 처리 속도 (FPS) |
| :--- | :--- | :--- | :--- | :--- |
| **1. FP32** | **11.88 ms** | **7.91 ms** | **37.56 ms** | **84.1 FPS** |
| **2. FP16** | **10.70 ms** | **7.13 ms** | **30.73 ms** | **93.5 FPS** |
| **3. INT8** | **10.84 ms** | **7.29 ms** | **30.92 ms** | **92.3 FPS** |

---

이번엔 C++ TensorRT 환경에서 YOLOv8n 모델의 정밀도별로 lock-free 방식으로 동영상 스트리밍 속도를 비교해 보았습니다.

본격적 추론하기 전에, Warm-up. 예열 과정 없이 추론했을 때는 max latency가 137.75ms에서 213.75 ms까지 웃도는 것을 확인하고, 

"아, warm-up 과정이 무조건 있어야, 정밀하게 평가가 가능하겠구나"라는 생각을 했습니다.

그러면.. 왜 Warm-up이 없으면 초반에 latency가 튀는 현상을 보이는가? 이유는 몇가지가 있습니다.

TensorRT 엔진 파일(.engine)을 메모리에 올린 직후, 첫 프레임이 GPU에 올라가는 순간 내부에서는 엄청난 양의 초기화 작업이 일어납니다.

먼저, GPU 내부 연산 코어들과 메모리 제어권을 초기화하고 연결하는 드라이버 연산이 최초 1회 작동하고, 
사용할 GPU 가상 메모리 주소를 매핑하는 부하가 발생합니다.

이러한 오버헤드가 첫 번째 프레임의 추론 시간에 고스란히 누적되기 때문입니다.

---

다시 본론으로 돌아와서, 이전에 했던 비동기 멀티스레드도 괜찮았지만, 그 방식은 producer가 프레임을 넣을 때(push) 문을 잠그고, 

consumer가 꺼낼 때(pop) 문을 잠급니다. 만약 그 타이밍이 겹치면 한쪽 스레드는 동작을 멈추고 기다려야 하는 병목이 생길 수 있습니다.

하지만 지금의 lock-free 구조는 std::atomic을 사용하여 mutex라는 자물쇠를 쓰지 않고도 스레드가 데이터를 안전하게 주고받을 수 있게 합니다.
스레드가 자물쇠 때문에 기다리는 시간이 0ms가 되기 때문에, 성능을 극한으로 끌어올릴 수 있습니다.

요약 표를 보시면, 이전 멀티스레드 mutex 방식에 비해 얼마나 성능 향상이 되었는지 확인할 수 있습니다.

멀티스레드 mutex 방식은 71.9 FPS가 맥시멈으로, 70대를 머무는 한계가 있었습니다.

👉 [멀티스레드 mutex 방식 보러 가기](https://github.com/wjdwogns2873-web/deep-learning-study/blob/main/09_%EB%B2%A4%EC%B9%98%EB%A7%88%ED%81%AC_%EB%B3%B4%EA%B3%A0%EC%84%9C/C%2B%2B_%EC%96%91%EC%9E%90%ED%99%94_%EB%AA%A8%EB%8D%B8_%EB%B9%84%EA%B5%90(%EB%B9%84%EB%8F%99%EA%B8%B0_%EB%A9%80%ED%8B%B0%EC%8A%A4%EB%A0%88%EB%93%9C).md)


하지만 지금 lock-free 방식은 모든 정밀도 엔진 면에서 mutex 방식을 압도합니다.

FP16 정밀도는 평균 FPS가 무려 93.5에 달합니다.

결과적으로 mutex 방식의 구조적 한계(타이밍이 겹치면 한쪽 스레드가 기다려야 하는 상황)를 극복하고, 성능을 극한으로 끌어올림으로서
C++ 설계의 중요성을 깨달았습니다.



---

<details>
<summary><b>선행작업</b></summary>

### `ONNX 파일 생성`
```python
model = YOLO('yolov8n.pt')

model.export(format='onnx')
if os.path.exists('yolov8n.onnx'):
    shutil.move('yolov8n.onnx', 'yolov8n.fp32.onnx')

model.export(format='onnx', half=True)
if os.path.exists('yolov8n.onnx'):
    shutil.move('yolov8n.onnx', 'yolov8n.fp16.onnx')

model.export(format='engine', int8=True)

# 라이브러리가 빌드 도중 생성한 TensorRT 호환용 INT8 ONNX 설계도 가로채기
if os.path.exists('yolov8n.onnx'):
    shutil.move('yolov8n.onnx', 'yolov8n.int8.onnx')
```

### `builder_all.cpp(engine 파일 생성)`
```cpp
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <NvInfer.h>
#include <NvOnnxParser.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

int main() {
    std::vector<std::string> models = { "yolov8n.fp32", "yolov8n.fp16", "yolov8n.int8" };

    for (const auto& modelName : models) {
        std::cout << "\n모델명: " << modelName << std::endl;

        // Builder 생성
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
        if (!builder) {
            std::cerr << "Builder 생성 실패: " << modelName << std::endl;
            continue;
        }

        // Network 생성
        nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
        if (!network) {
            std::cerr << "Network 생성 실패" << std::endl;
            delete builder;
            continue;
        }

        // ONNX Parser 생성 및 로드
        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
        std::string onnxPath = modelName + ".onnx";
        std::cout << onnxPath << " 생성중.." << std::endl;
        if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::cerr << "ONNX 파싱 실패" << onnxPath << std::endl;
            delete parser;
            delete network;
            delete builder;
            continue;
        }

        // Config 생성
        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
        
        // C++ TensorRT 직렬화 네트워크 빌드
        std::cout << "최적화 컴파일 가동중.." << std::endl;
        nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);

        if (!plan) {
            std::cerr << "최적화 엔진 변환 실패: " << modelName << std::endl;
            delete config;
            delete parser;
            delete network;
            delete builder;
            continue;
        }

        // 완제품 엔진 파일로 저장
        std::string enginePath = modelName + ".engine";
        std::ofstream engineFile(enginePath.c_str(), std::ios::binary);
        if (!engineFile) {
            std::cerr << "엔진 파일 생성 권한 오류: " << enginePath << std::endl;
            delete plan;
            delete config;
            delete parser;
            delete network;
            delete builder;
            continue;
        }

        engineFile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
        engineFile.close();

        std::cout << "저장 완료: " << enginePath << " (" << plan->size() << " bytes)" << std::endl;

        // 다음 모델을 위해 안전하게 자원 해제 (메모리 누수 방지)
        delete plan;
        delete config;
        delete parser;
        network->~INetworkDefinition(); // 최신 규격 안전 소멸자 호출
        delete builder;
    }

    std::cout << "\n모든 정밀도 엔진 빌드가 종료되었습니다" << std::endl;
    return 0;
}
```

### `engine 파일 빌드`
```python
!g++ -O3 -std=c++17 builder_all.cpp -o build_engine \
    -I/usr/include/x86_64-linux-gnu \
    -I/usr/local/cuda/include \
    -I/usr/include/opencv4 \
    -L/usr/lib/x86_64-linux-gnu \
    -L/usr/local/cuda/lib64 \
    -lnvinfer -lnvonnxparser -lcudart \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

!./build_engine
```


### `engine 파일 빌드 결과`
```text
모델명: yolov8n.fp32
yolov8n.fp32.onnx 생성중..
최적화 컴파일 가동중..
저장 완료: yolov8n.fp32.engine (172082820 bytes)

모델명: yolov8n.fp16
yolov8n.fp16.onnx 생성중..
최적화 컴파일 가동중..
저장 완료: yolov8n.fp16.engine (65151820 bytes)

모델명: yolov8n.int8
yolov8n.int8.onnx 생성중..
최적화 컴파일 가동중..
저장 완료: yolov8n.int8.engine (171826460 bytes)

모든 정밀도 엔진 빌드가 종료되었습니다
```
</details>

<details>
<summary><b>추론코드</b></summary>

### `YoloTRT.h`
```cpp
#ifndef YOLO_TRT_H
#define YOLO_TRT_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <filesystem>

// 검출 결과를 담는 구조체
struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

// TensorRT ILogger 로거
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class YoloTRT {
public:
    // 생성자 및 소멸자 (RAII 자원 관리)
    YoloTRT(const std::string& enginePath);
    ~YoloTRT();

    // End-to-End 검출 함수 (단순 Resize + 최적화 전/후처리)
    std::vector<Detection> detect(const cv::Mat& srcImg, float confThresh = 0.25f, float nmsThresh = 0.45f);

    // 순수 추론 전용 메서드
    void inferOnly();

private:
    TRTLogger gLogger;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    
    // 최적화된 내부 전처리/후처리 함수
    void preprocess(const cv::Mat& srcImg);
    std::vector<Detection> postprocess(int origW, int origH, float confThresh, float nmsThresh);

    cudaStream_t stream = nullptr;

    // GPU Device 버퍼
    void* d_input = nullptr;
    void* d_output = nullptr;

    // CPU Pinned Host 버퍼
    float* h_input = nullptr;
    float* h_output = nullptr;

    // 텐서 메모리 크기
    size_t inputSize = 1 * 3 * 640 * 640 * sizeof(float);
    size_t outputSize = 1 * 84 * 8400 * sizeof(float);

    // 후처리 버퍼 재사용 (동적 메모리 할당 해제 방지)
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    std::vector<int> indices;
};

#endif // YOLO_TRT_H
```

### `YoloTRT.cpp`
```cpp
#include "YoloTRT.h"
namespace fs = std::filesystem;

YoloTRT::YoloTRT(const std::string& enginePath) {
    // 1. 바이너리 엔진 파일 읽기
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "엔진 파일을 찾을 수 없습니다: " << enginePath << std::endl;
        return; 
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    // 2. TensorRT 런타임 및 세션 생성
    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engineData.data(), size);
    context = engine->createExecutionContext();

    // 3. CUDA 비동기 스트림 생성
    cudaStreamCreate(&stream);

    // 4. GPU Device 메모리 할당
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    // 5. 고속 DMA 전송을 위한 Pinned Memory (Page-locked) 할당
    cudaHostAlloc((void**)&h_input, inputSize, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_output, outputSize, cudaHostAllocDefault);

    // 6. 후처리 Vector 힙 재할당 오버헤드 방지를 위한 미리 용량 확보
    boxes.reserve(8400);
    confidences.reserve(8400);
    classIds.reserve(8400);
    indices.reserve(8400);

    // 7. 입출력 텐서 주소 바인딩
    const char* inputName = engine->getIOTensorName(0);
    const char* outputName = engine->getIOTensorName(1);
    context->setTensorAddress(inputName, d_input);
    context->setTensorAddress(outputName, d_output);
}

YoloTRT::~YoloTRT() {
    // RAII 기반 자동 자원 해제
    if (stream) cudaStreamDestroy(stream);
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);

    if (h_input) cudaFreeHost(h_input);
    if (h_output) cudaFreeHost(h_output);

    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
}

// 최적화 전처리: cv::split 생략 + Direct Pointer Write (HWC -> CHW)
void YoloTRT::preprocess(const cv::Mat& srcImg) {
    cv::Mat resizedImg;
    cv::resize(srcImg, resizedImg, cv::Size(640, 640));
    cv::cvtColor(resizedImg, resizedImg, cv::COLOR_BGR2RGB);

    cv::Mat floatImg;
    resizedImg.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    int channelSize = 640 * 640;
    float* data = reinterpret_cast<float*>(floatImg.data);

    // Pinned Memory 버퍼에 직접 포인터로 쓰기
    for (int i = 0; i < channelSize; ++i) {
        h_input[i]                   = data[i * 3 + 0]; // R
        h_input[i + channelSize]     = data[i * 3 + 1]; // G
        h_input[i + channelSize * 2] = data[i * 3 + 2]; // B
    }
}

// 최적화 후처리: 기존 Vector capacity 유지 후 clear() 재사용
std::vector<Detection> YoloTRT::postprocess(int origW, int origH, float confThresh, float nmsThresh) {
    boxes.clear();
    confidences.clear();
    classIds.clear();
    indices.clear();

    float scaleX = static_cast<float>(origW) / 640.0f;
    float scaleY = static_cast<float>(origH) / 640.0f;

    for (int i = 0; i < 8400; i++) {
        float cx = h_output[0 * 8400 + i];
        float cy = h_output[1 * 8400 + i];
        float w  = h_output[2 * 8400 + i];
        float h  = h_output[3 * 8400 + i];

        float maxScore = 0.0f;
        int maxClassId = -1;
        for (int c = 0; c < 80; c++) {
            float score = h_output[(c + 4) * 8400 + i];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }

        if (maxScore > confThresh) {
            float x1 = (cx - w / 2.0f) * scaleX;
            float y1 = (cy - h / 2.0f) * scaleY;
            float boxW = w * scaleX;
            float boxH = h * scaleY;

            int rectX = std::max(0, std::min(static_cast<int>(x1), origW - 1));
            int rectY = std::max(0, std::min(static_cast<int>(y1), origH - 1));
            int rectW = std::max(0, std::min(static_cast<int>(boxW), origW - rectX));
            int rectH = std::max(0, std::min(static_cast<int>(boxH), origH - rectY));

            boxes.emplace_back(rectX, rectY, rectW, rectH);
            confidences.emplace_back(maxScore);
            classIds.emplace_back(maxClassId);
        }
    }

    cv::dnn::NMSBoxes(boxes, confidences, confThresh, nmsThresh, indices);

    std::vector<Detection> detections;
    detections.reserve(indices.size());
    for (int idx : indices) {
        detections.push_back({ boxes[idx], confidences[idx], classIds[idx] });
    }

    return detections;
}

// End-to-End 전체 파이프라인 수행
std::vector<Detection> YoloTRT::detect(const cv::Mat& srcImg, float confThresh, float nmsThresh) {
    if (srcImg.empty()) return {};

    // 1. CPU 전처리
    preprocess(srcImg);

    // 2. 비동기 H2D 복사 -> GPU 추론 -> 비동기 D2H 복사
    cudaMemcpyAsync(d_input, h_input, inputSize, cudaMemcpyHostToDevice, stream);
    context->enqueueV3(stream);
    cudaMemcpyAsync(h_output, d_output, outputSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 3. CPU 후처리 (NMS)
    return postprocess(srcImg.cols, srcImg.rows, confThresh, nmsThresh);
}

// 순수 추론 수행 메서드
void YoloTRT::inferOnly() {
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
}
```

### `main_lockfree.cpp`
```cpp
#include "YoloTRT.h"
#include <chrono>
#include <numeric>
#include <vector>
#include <iostream>
#include <algorithm>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>

// std::mutex를 하나도 쓰지 않는 고속 lock-free 순환 큐
class LockFreeFrameQueue {
private:
    static constexpr size_t kCapacity = 32;
    cv::Mat ringBuffer[kCapacity];

    // 원자적 인덱스 포인터 (CPU 하드웨어 레벨에서 직접 제어)
    std::atomic<size_t> writeIndex{0};
    std::atomic<size_t> readIndex{0};

public:
    LockFreeFrameQueue() {
        // 메모리 풀링: 시작할 때 640 640 크기 방 32개를 미리 파둡니다. (동적 할당 배제)
        for (size_t i = 0; i < kCapacity; i++) {
            ringBuffer[i] = cv::Mat::zeros(640, 640, CV_8UC3);
        }
    }

    // 생상자용: 자물쇠 없이 원자적 연산으로 큐에 삽입
    bool push(const cv::Mat& frame) {
        size_t currentWrite = writeIndex.load(std::memory_order_relaxed);
        size_t currentRead = readIndex.load(std::memory_order_acquire);

        // 큐가 가득 찼는지 검사 (원자적 인덱스 간격 확인)
        if ((currentWrite - currentRead) >= kCapacity) {
            return false; // 자물쇠가 없으므로 스레드를 재우지 않고 즉시 false 반환
        }

        // 복사 오버헤드 최소화: 미리 파둔 방의 메모리에 픽셀 그림만 덮어씁니다. clone 생략
        size_t index = currentWrite % kCapacity;
        frame.copyTo(ringBuffer[index]);

        // 쓰기 완료 신호를 원자적으로 업데이트
        writeIndex.store(currentWrite + 1, std::memory_order_release);
        return true;
    }

    // 소비자용: 자물쇠 없이 원자적 연산으로 큐에서 추출
    bool pop(cv::Mat& frame, const std::atomic<bool>& stopSignal) {
        while (true) {
            size_t currentRead = readIndex.load(std::memory_order_relaxed);
            size_t currentWrite = writeIndex.load(std::memory_order_acquire);

            // 큐가 비어있는 상태 체크
            if (currentRead == currentWrite) {
                if (stopSignal.load(std::memory_order_relaxed)) {
                    return false; // 동영상 종료 및 큐 고갈 시 루프 종료
                }
                // Budy Waiting을 방지하고 CPU를 잠시 양보(yield)
                std::this_thread::yield();
                continue;
            }

            // 안전하게 데이터를 꺼내 바구니에 담기
            size_t index = currentRead % kCapacity;
            frame = ringBuffer[index];

            // 읽기 완료 신호를 원자적으로 업데이트 (생상자에게 빈자리 알림)
            readIndex.store(currentRead + 1, std::memory_order_release);
            return true;
        }
    }
};

// 스레드 1: 고속 비디오 디코딩 스레드 루프
void videoDecodingThread(const std::string& videoPath, LockFreeFrameQueue& lfQueue, std::atomic<bool>& stopSignal) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "비디오 파일을 열 수 없습니다: " << videoPath << std::endl;
        stopSignal = true;
        return;
    }

    cv::Mat frame;
    while (!stopSignal) {
        cap >> frame;
        if (frame.empty()) break;

        // 큐가 가득 차서 push가 실패하면, 자리가 생길 때까지 무한루프로 대기
        while (!lfQueue.push(frame) && !stopSignal) {
            std::this_thread::yield();
        }
    }
    cap.release();
    stopSignal = true;
}

int main() {
    std::string engineName = "yolov8n.fp32.engine";

    YoloTRT detector(engineName);
    std::string videoPath = "./test_videos/sample.mp4";

    cv::VideoCapture tempCap(videoPath);
    int totalFrames = static_cast<int>(tempCap.get(cv::CAP_PROP_FRAME_COUNT));
    tempCap.release();

    std::cout << "warm up start" << std::endl;
    cv::Mat dummyFrame = cv::Mat::zeros(640, 640, CV_8UC3);
    for (int i = 0; i < 50; i++) {
        detector.detect(dummyFrame, 0.25f, 0.45f);
    }
    std::cout << "warm up end" << std::endl;

    LockFreeFrameQueue lfQueue;
    std::atomic<bool> stopSignal(false);
    std::vector<double> latencies;
    latencies.reserve(totalFrames);

    // 스레드 가동
    std::thread reader(videoDecodingThread, videoPath, std::ref(lfQueue), std::ref(stopSignal));

    int frameCount = 0;
    while (true) {
        cv::Mat frame;
        if (!lfQueue.pop(frame, stopSignal)) {
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();

        auto detections = detector.detect(frame, 0.25f, 0.45f);
        for (const auto& det : detections) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(ms);
        frameCount++;
    }

    if (reader.joinable()) {
        reader.join();
    }

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();
    double min = *std::min_element(latencies.begin(), latencies.end());
    double max = *std::max_element(latencies.begin(), latencies.end());

    std::cout << "================ [lock-free 큐 고속 비동기 TRT Video 벤치마크] ================" << std::endl;
    std::cout << "사용한 가속 엔진: " << engineName << std::endl;
    std::cout << "처리 프레임 수: " << frameCount << " / " << totalFrames << std::endl;
    std::cout << "순수 추론 파이프라인 Latency (Avg): " << cv::format("%.2f", avg) << " ms" << std::endl;
    std::cout << "최소 Latency (Min): " << cv::format("%.2f", min) << " ms" << std::endl;
    std::cout << "최대 Latency (Max): " << cv::format("%.2f", max) << " ms" << std::endl;
    std::cout << "최종 lock-free 파이프라인 처리 속도: " << cv::format("%.1f", 1000.0 / avg) << " FPS" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    return 0;
}
```
</details>

<details>
<summary><b>추론 시작 및 추론 결과</b></summary>

### 
```python
!g++ -O3 -std=c++17 main_lockfree.cpp YoloTRT.cpp -o streaming_test \
    -I/usr/include/x86_64-linux-gnu \
    -I/usr/local/cuda/include \
    -I/usr/include/opencv4 \
    -L/usr/lib/x86_64-linux-gnu \
    -L/usr/local/cuda/lib64 \
    -lnvinfer -lnvonnxparser -lcudart \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_dnn -lpthread

!./streaming_test
```


### `FP32 결과`
```content
[멀티쓰레드 벤치마크] 로드 중인 엔진: yolov8n.fp32.engine
멀티 쓰레드 파이프라인 가동... 추론 측정 시작합니다.

================ [lock-free 큐 고속 비동기 TRT Video 벤치마크] ================
사용한 가속 엔진: yolov8n.fp32.engine
처리 프레임 수: 3129 / 3129
순수 추론 파이프라인 Latency (Avg): 11.88 ms
최소 Latency (Min): 7.91 ms
최대 Latency (Max): 37.56 ms
최종 lock-free 파이프라인 처리 속도: 84.1 FPS
=========================================================================
```

### `FP16 결과`
```content
[멀티쓰레드 벤치마크] 로드 중인 엔진: yolov8n.fp16.engine
멀티 쓰레드 파이프라인 가동... 추론 측정 시작합니다.

================ [lock-free 큐 고속 비동기 TRT Video 벤치마크] ================
사용한 가속 엔진: yolov8n.fp16.engine
처리 프레임 수: 3129 / 3129
순수 추론 파이프라인 Latency (Avg): 10.70 ms
최소 Latency (Min): 7.13 ms
최대 Latency (Max): 30.73 ms
최종 lock-free 파이프라인 처리 속도: 93.5 FPS
=========================================================================
```

### `INT8 결과`
```content
[멀티쓰레드 벤치마크] 로드 중인 엔진: yolov8n.int8.engine
멀티 쓰레드 파이프라인 가동... 추론 측정 시작합니다.

================ [lock-free 큐 고속 비동기 TRT Video 벤치마크] ================
사용한 가속 엔진: yolov8n.int8.engine
처리 프레임 수: 3129 / 3129
순수 추론 파이프라인 Latency (Avg): 10.84 ms
최소 Latency (Min): 7.29 ms
최대 Latency (Max): 30.92 ms
최종 lock-free 파이프라인 처리 속도: 92.3 FPS
=========================================================================
```

</details>
