# C++ TensorRT: FP32 vs FP16 vs INT8 동영상 스트리밍 벤치마크 보고서 (비동기 멀티스레드 방식)


## 📊 종합 성능 비교 요약

| 정밀도 엔진 | 평균 Latency (Avg) | 최소 Latency (Min) | 최대 Latency (Max) | 평균 처리 속도 (FPS) |
| :--- | :--- | :--- | :--- | :--- |
| **1. FP32** | **14.99 ms** | **8.81 ms** | **42.65 ms** | **66.7 FPS** |
| **2. FP16** | **13.91 ms** | **7.31 ms** | **37.30 ms** | **71.9 FPS** |
| **3. INT8** | **14.37 ms** | **7.68 ms** | **40.04 ms** | **69.6 FPS** |

---

이번엔 C++ TensorRT 환경에서 YOLOv8n 모델의 정밀도별로 비동기 멀티스레드 방식으로 동영상 스트리밍시 성능을 비교해 보았습니다.

이전에 했던 실험인 싱글스레드 방식과는 선행작업은 동일합니다.

하지만 main.cpp가 main_multi.cpp로 바뀌었습니다.

그리고 추론하기 위해 빌드하는 과정은 동일합니다.

FP16 모델이 71.9FPS로 가장 빨랐습니다.
T4 GPU 환경에서 FP16은 형변환 오버헤드가 없으면서도 하드웨어 연산 효율이 좋기 때문에 가장 빨랐습니다.

이론상 가장 빨라야 할 INT8이 오히려 FP16보다 아주 미세하게 느려졌습니다.
싱글스레드 때와 마찬가지로 Quantize/Dequantize 변환 노드 추가로 인해 약간의 오버헤드가 발생했습니다.

그러면 여기서 의문이 생깁니다.
싱글스레드는 90~100FPS 수준이 나왔는데 비동기 멀티스레드는 70FPS에 불과한가요?

이 현상은 멀티스레드가 느려진 것이 아니라, 기존 싱글스레드 환경의 구조적 한계로 인해 발생했던 측정 오차가 있었는데, 
이것이 멀티스레드로 전환하면서 정상화된 결과입니다.

싱글스레드의 100FPS는 프레임 누락일 확률이 높습니다.
싱글 스레드 동기식 루프에서는 CPU가 이미지를 디코딩하고 가공하는 연산속도가 조금이라도 지연되면,
OpenCV 비디오 백엔드가 이 타임라인을 맞추기 위해 중간 프레임들을 통째로 누락시키거나 디코딩을 생략합니다.
즉, 전체 3129장의 프레임 처리 중에 중간 몇 프레임들은 건너뛰어서 FPS 숫자만 높게 찍혔을 확률이 높습니다.

하지만 비동기 멀티스레드의 70FPS는 전체 3129장을 처리한 성능입니다.
소스 코드를 보면, 디코딩 스레드가 백그라운드에서 원본 동영상의 모든 프레임을 큐에 적재합니다.
메인 스레드는 큐에 쌓인 3129장의 프레임을 정직하게 전부 처리하고 후처리도 하기 때문에, 
한계 성능인 71.9FPS가 나온 것입니다.

근데, 비동기 구조로 CPU 대기 시간을 없앴음에도 70FPS 부근에 수렴하는 이유는, 
매 프레임마다 대용량 픽셀 데이터를 CPU 메모리에서 GPU 메모리로 옮기고(Host to Device) 다시 받아오는
PCIe 버스의 물리적인 전송 대역폭 비용이 약 13~14ms의 고정 상수로 버티고 있기 때문입니다.

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

### `main_multi.cpp`
```cpp
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
```
</details>

<details>
<summary><b>추론 시작 및 추론 결과</b></summary>

### 
```python
!g++ -O3 -std=c++17 main_multi.cpp YoloTRT.cpp -o streaming_test \
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

================ [비동기 멀티쓰레드 TRT Video 벤치마크] ================
사용한 가속 엔진: yolov8n.fp32.engine
처리 프레임 수: 3129 / 3129
순수 추론 파이프라인 Latency (Avg): 14.99 ms
최소 Latency (Min): 8.81 ms
최대 Latency (Max): 42.65 ms
최종 동영상 처리 속도: 66.7 FPS
=========================================================================
```

### `FP16 결과`
```content
[멀티쓰레드 벤치마크] 로드 중인 엔진: yolov8n.fp16.engine
멀티 쓰레드 파이프라인 가동... 추론 측정 시작합니다.

================ [비동기 멀티쓰레드 TRT Video 벤치마크] ================
사용한 가속 엔진: yolov8n.fp16.engine
처리 프레임 수: 3129 / 3129
순수 추론 파이프라인 Latency (Avg): 13.91 ms
최소 Latency (Min): 7.31 ms
최대 Latency (Max): 37.30 ms
최종 동영상 처리 속도: 71.9 FPS
=========================================================================
```

### `INT8 결과`
```content
[멀티쓰레드 벤치마크] 로드 중인 엔진: yolov8n.int8.engine
멀티 쓰레드 파이프라인 가동... 추론 측정 시작합니다.

================ [비동기 멀티쓰레드 TRT Video 벤치마크] ================
사용한 가속 엔진: yolov8n.int8.engine
처리 프레임 수: 3129 / 3129
순수 추론 파이프라인 Latency (Avg): 14.37 ms
최소 Latency (Min): 7.68 ms
최대 Latency (Max): 40.04 ms
최종 동영상 처리 속도: 69.6 FPS
=========================================================================
```

</details>
