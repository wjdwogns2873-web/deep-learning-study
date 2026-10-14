# C++ TensorRT: FP32 vs FP16 vs INT8 동영상 스트리밍 벤치마크 보고서


## 📊 종합 성능 비교 요약

| 정밀도 엔진 | 평균 Latency (Avg) | 최소 Latency (Min) | 최대 Latency (Max) | 평균 처리 속도 (FPS) |
| :--- | :--- | :--- | :--- | :--- |
| **1. FP32** | **9.91 ms** | **6.82 ms** | **24.70 ms** | **100.9 FPS** |
| **2. FP16** | **8.70 ms** | **6.00 ms** | **19.74 ms** | **115.0 FPS** |
| **3. INT8** | **10.11 ms** | **6.92 ms** | **20.76 ms** | **98.9 FPS** |

---

C++ TensorRT 환경에서 YOLOv8n 모델의 정밀도별로 동영상 스트리밍시 성능을 비교해보고 싶어서 해본 실험입니다. 

양자화 기법 중에 FP16은 몇번 해봤지만, INT8은 이번에 처음 도전해봤습니다. 

양자화라는 게 모델의 가중치의 데이터타입을 비트 수가 작은 타입으로 변환하는 걸 의미하는데, 양자화를 하면 연산량이 줄어든다고 알고 있었습니다.

그래서 저는 INT8 > FP16 > FP32 순으로 빠를 거라고 예상했지만, 실제로 제가 예상한 결과와는 조금 달랐습니다. 그 이유를 파헤쳐 보았습니다.

1. FP16이 가장 빨랐던 이유
   
T4 GPU는 반정밀도 실수를 계산하는 FP16 연산 유닛이 최적화되어 있습니다.
가중치 데이터만 16비트로 압축하기 때문에 연산 도중 정밀도를 변환하기 위한 형변환 오버헤드가 발생하지 않아서 빨랐습니다.

3. INT8이 FP32와 비슷한 수준을 보여줬던 이유
   
TensorRT 11.x 환경에서는 정확도 손실을 막기 위해 그래프 내부에 Quantize/Dequantize 변환 노드 설계도를 강제로 주입합니다.
이로 인해 GPU 내부에서 레이어를 거칠 때마다 [FP32 -> INT8 -> 연산 -> FP32] 라는 데이터 재가공 과정이 매 프레임마다 누적됩니다.
따라서 가중치 압축으로 얻은 이득보다 타입 변환 제어용 레이어 구조물들이 발생시키는 오버헤드가 더 커졌기 때문에 이러한 결과가 나왔습니다.


## 하지만 현업에서는 단순히 FPS가 더 높은 모델을 채택하는 건 아닙니다.

먼저, FP16을 가장 많이 선호합니다.

FP32에 비해 정확도 손실이 0%에 수렴하면서도, 모델 용량과 VRAM 점유율을 절반으로 감소시킵니다.
그리고 최신 가속 엔진과의 호환성이 좋아서 버전 문제나 컴파일 에러가 없는 이상 최고의 안정성을 보장하기 때문에 현업에서는 국룰로 자리잡았습니다.

두 번째로 INT8을 선호합니다.

자율주행 차량용 칩이나 스마트폰, 임베디드 장비처럼 하드웨어 자원과 메모리가 극도로 제한된 엣지 디바이스 환경에서 무조건 구겨넣어야 할 때 사용합니다.

그리고 FP32는 실무에서 절대로 쓰지 않습니다.

연구실에서 모델을 학습시키거나 분석할 때만 사용하고 GPU VRAM을 많이 잡아먹어서 실제 실무 환경에서는 절대로 쓰지 않습니다.

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

### `main.cpp`
```cpp
#include "YoloTRT.h"
#include <chrono>
#include <numeric>
#include <vector>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

int main() {
    // 1. C++ TensorRT 엔진 로드 (원하는 모델에 따라서 선택)
    std::string modelName = "yolov8n.fp32.engine";
    // std::string modelName = "yolov8n.fp16.engine";
    // std::string modelName = "yolov8n.int8.engine";
    YoloTRT detector(modelName);

    std::cout << modelName << " 엔진으로 비디오 추론 시작" << std::endl;

    // 2. 비디오 파일 열기
    std::string videoPath = "./test_videos/sample.mp4";
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "비디오 파일을 열 수 없습니다: " << videoPath << std::endl;
        return -1;
    }

    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double inputFps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "입력 비디오 정보: " << totalFrames << " Frames, " << inputFps << " FPS" << std::endl;

    // 3. Warm-up (첫 프레임 기반 50회 예열)
    cv::Mat frame;
    cap >> frame;
    if (!frame.empty()) {
        for (int i = 0; i < 50; ++i) {
            detector.detect(frame, 0.25f, 0.45f);
        }
    }

    // 비디오 읽기 위치를 처음 프레임으로 리셋
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    std::vector<double> latencies;
    latencies.reserve(totalFrames);
    int frameCount = 0;

    // 4. 동영상 프레임 단위 End-to-End 측정 (디코딩 + 전처리 + H2D + 추론 + D2H + NMS + 시각화)
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        auto start = std::chrono::high_resolution_clock::now();

        // 전처리 -> TRT 추론 -> NMS 후처리
        auto detections = detector.detect(frame, 0.25f, 0.45f);

        // 검출 결과 바운딩 박스 렌더링
        for (const auto& det : detections) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(ms);
        frameCount++;
    }

    cap.release();

    if (latencies.empty()) {
        std::cerr << "처리된 프레임이 없습니다." << std::endl;
        return -1;
    }

    // 5. 통계 산출
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();
    double min = *std::min_element(latencies.begin(), latencies.end());
    double max = *std::max_element(latencies.begin(), latencies.end());

    std::cout << "==== C++ Native " << modelName << " 엔진 동영상 스트리밍 벤치마크 ====" << std::endl;
    std::cout << "처리 프레임 수: " << frameCount << " / " << totalFrames << std::endl;
    std::cout << "평균 Latency (Avg): " << cv::format("%.2f", avg) << " ms" << std::endl;
    std::cout << "최소 Latency (Min): " << cv::format("%.2f", min) << " ms" << std::endl;
    std::cout << "최대 Latency (Max): " << cv::format("%.2f", max) << " ms" << std::endl;
    std::cout << "평균 동영상 처리 FPS: " << cv::format("%.1f", 1000.0 / avg) << " FPS" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    return 0;
}
```
</details>

<details>
<summary><b>추론 시작 및 추론 결과</b></summary>

### 
```python
!g++ -O3 -std=c++17 main.cpp YoloTRT.cpp -o streaming_test \
    -I/usr/include/x86_64-linux-gnu \
    -I/usr/local/cuda/include \
    -I/usr/include/opencv4 \
    -L/usr/lib/x86_64-linux-gnu \
    -L/usr/local/cuda/lib64 \
    -lnvinfer -lnvonnxparser -lcudart \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_dnn

!./streaming_test
```


### `FP32 결과`
```content
yolov8n.fp32.engine 엔진으로 비디오 추론 시작
입력 비디오 정보: 3129 Frames, 25 FPS
==== C++ Native yolov8n.fp32.engine 엔진 동영상 스트리밍 벤치마크 ====
처리 프레임 수: 3129 / 3129
평균 Latency (Avg): 9.91 ms
최소 Latency (Min): 6.82 ms
최대 Latency (Max): 24.70 ms
평균 동영상 처리 FPS: 100.9 FPS
=========================================================================
```

### `FP16 결과`
```content
yolov8n.fp16.engine 엔진으로 비디오 추론 시작
입력 비디오 정보: 3129 Frames, 25 FPS
==== C++ Native yolov8n.fp16.engine 엔진 동영상 스트리밍 벤치마크 ====
처리 프레임 수: 3129 / 3129
평균 Latency (Avg): 8.70 ms
최소 Latency (Min): 6.00 ms
최대 Latency (Max): 19.74 ms
평균 동영상 처리 FPS: 115.0 FPS
=========================================================================
```

### `INT8 결과`
```content
yolov8n.int8.engine 엔진으로 비디오 추론 시작
입력 비디오 정보: 3129 Frames, 25 FPS
==== C++ Native yolov8n.int8.engine 엔진 동영상 스트리밍 벤치마크 ====
처리 프레임 수: 3129 / 3129
평균 Latency (Avg): 10.11 ms
최소 Latency (Min): 6.92 ms
최대 Latency (Max): 20.76 ms
평균 동영상 처리 FPS: 98.9 FPS
=========================================================================
```

</details>
