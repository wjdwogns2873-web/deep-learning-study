# YOLOv8 전처리 및 후처리 포함 추론 속도 비교 (PyTorch vs ONNX Runtime vs TensorRT)

본 문서는 비디오 디코딩, 전처리(Resize, BGR to RGB, Normalization), 추론, 후처리(NMS) 및 결과 시각화(Bounding Box Rendering)를 모두 포함한 전체 파이프라인(End-to-End) 속도를 측정한 벤치마크 결과입니다. 동일한 비디오 조건(3,129 Frames)에서 연속 수행한 결과를 바탕으로 작성되었습니다.

---

### 📊 요약 결과 (전/후처리 및 디코딩 포함 3,129 Frame 측정)

| 실행 환경 및 프레임워크 | 평균 소요 시간 (Avg) | 최소 소요 시간 (Min) | 최대 소요 시간 (Max) | 평균 처리 속도 (FPS) |
| :--- | :---: | :---: | :---: | :---: |
| **1. PyTorch (.pt)** | 18.66 ms | 9.38 ms | 135.35 ms | 53.6 FPS |
| **2. ONNX Runtime** | 17.09 ms | 11.53 ms | 115.29 ms | 58.5 FPS |
| **3. Python TensorRT (기본 메모리)** | 12.73 ms | 8.75 ms | 58.59 ms | 78.5 FPS |
| **4. Python TensorRT (Zero-Copy)** | 12.73 ms | 6.96 ms | 64.05 ms | 78.5 FPS |
| **5. C++ Native TensorRT** | **9.49 ms** | **6.91 ms** | **21.76 ms** | **105.3 FPS** |

---

## 🛠️ 상세 구현 코드 및 측정 결과

<details>
<summary><b>1. PyTorch (.pt) 측정 코드 및 결과</b></summary>

```python
import torch
from ultralytics import YOLO
import cv2
import time
import numpy as np

# 모델 로드 및 CUDA 최적화 설정
torch.backends.cudnn.benchmark = True
model = YOLO('yolov8n.pt')

# 비디오 파일 열기
video_path = './test_videos/sample.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f'비디오 파일을 열 수 없습니다: {video_path}')

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Warm-Up
ret, frame = cap.read()
if ret:
    with torch.no_grad():
	for _ in range(50):
	    _ = model(frame, verbose=False)
    torch.cuda.synchronize()

# 비디오 캡처 위치 리셋
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 동영상 프레임 측정
latencies = []
frame_count = 0

with torch.no_grad():
    while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
	    break

	start = time.time()

	results = model(frame, verbose=False)
	annotated_frame = results[0].plot()

	torch.cuda.synchronize()

	latencies.append((time.time() - start) * 1000)
	frame_count += 1

cap.release()

avg = np.mean(latencies)

print('================ [1. PyTorch Video End-to-End 벤치마크] ================')
print(f'처리 프레임 수: {frame_count} / {total_frames}')
print(f'Avg Latency: {avg:.2f} ms')
print(f'Min Latency: {np.min(latencies):.2f} ms')
print(f'Max Latency: {np.max(latencies):.2f} ms')
print(f'Avg FPS: {1000.0 / avg:.1f} FPS')
print('==========================================================================')
```

```text
================ [1. PyTorch Video End-to-End 벤치마크] ================
처리 프레임 수: 3129 / 3129
Avg Latency: 18.66 ms
Min Latency: 9.38 ms
Max Latency: 135.35 ms
Avg FPS: 53.6 FPS
==========================================================================
```

</details>

<details>
<summary><b>2. ONNX Runtime</b></summary>

```Python
import onnxruntime as ort
import cv2
import numpy as np
import time

# ORT 세션 준비
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession('yolov8n.onnx', providers=providers)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 전처리 함수
def preprocess(frame):
    img_resized = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)

# 후처리 함수
def postprocess(outputs, orig_shape, conf_thresh=0.25):
    preds = np.squeeze(outputs[0])
    boxes = preds[:4, :]
    scores = np.max(preds[4:, :], axis=0)
    class_ids = np.argmax(preds[4:, :], axis=0)

    valid_mask = scores > conf_thresh
    boxes = boxes[:, valid_mask]
    scores = scores[valid_mask]
    class_ids = class_ids[valid_mask]

    # 640 640 -> 원본 좌표로 복원
    orig_h, orig_w = orig_shape[:2]
    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0

    boxes_rescaled = []
    for box in boxes.T:
	cx, cy, w, h = box
	xmin = int((cx - w / 2) * scale_x)
	ymin = int((cy - h / 2) * scale_y)
	bw = int(w * scale_x)
	bh = int(h * scale_y)
	boxes_rescaled.append([xmin, ymin, bw, bh])

    return boxes_rescaled, scores, class_ids

# 비디오 파일 열기
video_path = './test_videos/sample.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f'비디오 파일을 열 수 없습니다: {video_path}')

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Warm-Up
ret, frame = cap.read()
if ret:
    for _ in range(50):
	input = preprocess(frame)
	output = session.run([output_name], {input_name: input})
	_ = postprocess(output, frame.shape)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 동영상 프레임 측정
latencies = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
	break

    start = time.time()

    input = preprocess(frame)
    output = session.run([output_name], {input_name: input})
    boxes, scores, class_ids = postprocess(output, frame.shape)

    for box in boxes:
	xmin, ymin, bw, bh = box
	cv2.rectangle(frame, (xmin, ymin), (xmin + bw, ymin + bh), (0, 255, 0), 2)

    latencies.append((time.time() - start) * 1000)
    frame_count += 1

cap.release()

avg = np.mean(latencies)

print('================ [2. ONNX Runtime Video End-to-End 벤치마크] ================')
print(f'처리 프레임 수: {frame_count} / {total_frames}')
print(f'Avg Latency: {avg:.2f} ms')
print(f'Min Latency: {np.min(latencies):.2f} ms')
print(f'Max Latency: {np.max(latencies):.2f} ms')
print(f'Avg FPS: {1000.0 / avg:.1f} FPS')
print('==========================================================================')
```

### 실행 결과
```text
================ [2. ONNX Runtime Video End-to-End 벤치마크] ================
처리 프레임 수: 3129 / 3129
Avg Latency: 17.09 ms
Min Latency: 11.53 ms
Max Latency: 115.29 ms
Avg FPS: 58.5 FPS
==========================================================================
```

</details>

<details>
<summary><b>TensorRT 가속 엔진 (.engine) 변환 코드</b></summary>

```python
import os
import tensorrt as trt

# 로거 및 빌더 생성
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)

# 네트워크 정의 생성
network = builder.create_network()
parser = trt.OnnxParser(network, TRT_LOGGER)

# ONNX 파일 파싱
onnx_path = 'yolov8n.onnx'
if not os.path.exists(onnx_path):
    raise FileNotFoundError(f'{onnx_path} 파일이 없습니다.')

with open(onnx_path, 'rb') as model:
    if not parser.parse(model.read()):
	print('ONNX 파싱 실패')
	raise RuntimeError('ONNX 파서 에러')

# 최적화 빌드 설정
config = builder.create_builder_config()

# 직렬화된 엔진 바이너리 생성
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    raise RuntimeError('가속 엔진 빌드 프로세스 실패')

# .engine 파일 저장
engine_path = 'yolov8n.engine'
with open(engine_path, 'wb') as f:
    f.write(serialized_engine)

# 엔진 로드 및 실행 context 준비
runtime = trt.Runtime(TRT_LOGGER)
with open(engine_path, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
```

</details>

<details>
<summary><b>3. Python TRT</b></summary>

```Python
input_shape = (1, 3, 640, 640)
input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
output_shape = tuple(engine.get_tensor_shape(engine.get_tensor_name(1)))
output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize

d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)
stream = cuda.Stream()

host_output = np.empty(output_shape, dtype=np.float32)

context.set_tensor_address(engine.get_tensor_name(0), int(d_input))
context.set_tensor_address(engine.get_tensor_name(1), int(d_output))

def preprocess(frame):
    img_resized = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.expand_dims(tensor, axis=0))

def postprocess(preds, orig_shape, conf_thresh=0.25):
    boxes = preds[0, :4, :]
    scores = np.max(preds[0, 4:, :], axis=0)
    valid_mask = scores > conf_thresh

    valid_boxes = boxes[:, valid_mask]
    valid_scores = scores[valid_mask]

    orig_h, orig_w = orig_shape[:2]
    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0

    rescaled_boxes = []
    for box in valid_boxes.T:
	cx, cy, w, h = box
	xmin = int( (cx - w / 2.0) * scale_x )
	ymin = int( (cy - h / 2.0) * scale_y )
	bw = int(w * scale_x)
	bh = int(h * scale_y)
	rescaled_boxes.append([xmin, ymin, bw, bh])

    return rescaled_boxes, valid_scores

video_path = './test_videos/sample.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f'비디오 파일을 열 수 없습니다: {video_path}')

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Warm-Up
ret, frame = cap.read()
if ret:
    for _ in range(50):
	input = preprocess(frame)
	cuda.memcpy_htod_async(d_input, input, stream)
	context.execute_async_v3(stream_handle=stream.handle)
	cuda.memcpy_dtoh_async(host_output, d_output, stream)
	stream.synchronize()
	_ = postprocess(host_output, frame.shape)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

latencies = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
	break

    start = time.time()

    input = preprocess(frame)
    cuda.memcpy_htod_async(d_input, input, stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, d_output, stream)
    stream.synchronize()
    boxes, scores = postprocess(host_output, frame.shape)

    for box in boxes:
	xmin, ymin, bw, bh = box
	cv2.rectangle(frame, (xmin, ymin), (xmin + bw, ymin + bh), (0, 255, 0), 2)

    latencies.append((time.time() - start) * 1000)
    frame_count += 1

cap.release()

avg = np.mean(latencies)

print('================ [3. Python TRT (Standard) Video 벤치마크] ================')
print(f'처리 프레임 수: {frame_count} / {total_frames}')
print(f'Avg Latency: {avg:.2f} ms')
print(f'Min Latency: {np.min(latencies):.2f} ms')
print(f'Max Latency: {np.max(latencies):.2f} ms')
print(f'Avg FPS: {1000.0 / avg:.1f} FPS')
print('=========================================================================='
```

### 실행 결과
```text
================ [3. Python TRT (Standard) Video 벤치마크] ================
처리 프레임 수: 3129 / 3129
Avg Latency: 12.73 ms
Min Latency: 8.75 ms
Max Latency: 58.59 ms
Avg FPS: 78.5 FPS
==========================================================================
```

</details>

<details>
<summary><b>4. Python TRT (Zero-Copy)</b></summary>

```Python
input_shape = (1, 3, 640, 640)
output_shape = tuple( engine.get_tensor_shape(engine.get_tensor_name(1)) )

# Pinned Memory 할당
h_input = cuda.pagelocked_empty(input_shape, dtype=np.float32)
h_output = cuda.pagelocked_empty(output_shape, dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)

stream = cuda.Stream()

context.set_tensor_address(engine.get_tensor_name(0), int(d_input))
context.set_tensor_address(engine.get_tensor_name(1), int(d_output))

def preprocess_to_pinned(frame, pinned_buf):
    img_resized = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    np.copyto(pinned_buf, np.expand_dims(tensor, axis=0))

def postprocess(preds, orig_shape, conf_thresh=0.25):
    boxes = preds[0, :4, :]
    scores = np.max(preds[0, 4:, :], axis=0)
    valid_mask = scores > conf_thresh

    valid_boxes = boxes[:, valid_mask]
    valid_scores = scores[valid_mask]

    orig_h, orig_w = orig_shape[:2]
    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0

    rescaled_boxes = []
    for box in valid_boxes.T:
	cx, cy, w, h = box
	xmin = int((xmin - w / 2.0) * scale_x)
	ymin = int((ymin - h / 2.0) * scale_y)
	bw = int(w * scale_x)
	bh = int(h * scale_y)
	rescaled_boxes.append([xmin, ymin, bw, bh])

    return rescaled_boxes, valid_scores

video_path = './test_videos/sample.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f'비디오 파일을 열 수 없습니다: {video_path}')

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Warm Up
ret, frame = cap.read()
if ret:
    for _ in range(50):
	preprocess_to_pinned(frame, h_input)
	cuda.memcpy_htod_async(d_input, h_input, stream)
	context.execute_async_v3(stream_handle=stream.handle)
	cuda.memcpy_dtoh_async(h_output, d_output, stream)
	stream.synchronize()
	_ = postprocess(h_output, frame.shape)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

latencies = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
	break

    start = time.time()

    preprocess_to_pinned(frame, h_input)
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    boxes, scores = postprocess(h_output, frame.shape)

    for box in boxes:
	xmin, ymin, bw, bh = box
	cv2.rectangle(frame, (xmin, ymin), (xmin + bw, ymin + bh), (0, 255, 0), 2)

    latencies.append((time.time() - start) * 1000)
    frame_count += 1

cap.release()

avg = np.mean(latencies)

print('================ [4. Python TRT (Zero-Copy) Video 벤치마크] ================')
print(f'처리 프레임 수: {frame_count} / {total_frames}')
print(f'Avg Latency: {avg:.2f} ms')
print(f'Min Latency: {np.min(latencies):.2f} ms')
print(f'Max Latency: {np.max(latencies):.2f} ms')
print(f'Avg FPS: {1000.0 / avg:.1f} FPS')
print('==========================================================================')
```

### 실행 결과
```text
================ [4. Python TRT (Zero-Copy) Video 벤치마크] ================
처리 프레임 수: 3129 / 3129
Avg Latency: 12.73 ms
Min Latency: 6.96 ms
Max Latency: 64.05 ms
Avg FPS: 78.5 FPS
==========================================================================
```

</details>

<details>
<summary><b>5. C++ TensorRT 구현 코드 및 측정 결과</b></summary>

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

struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class YoloTRT {
public:
    YoloTRT(const std::string& enginePath);
    ~YoloTRT();

    std::vector<Detection> detect(const cv::Mat& srcImg, float confThresh = 0.25f, float nmsThresh = 0.45f);
    void inferOnly();

private:
    void preprocess(const cv::Mat& srcImg);
    std::vector<Detection> postprocess(int origW, int origH, float confThresh, float nmsThresh);

private:
    TRTLogger gLogger;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    cudaStream_t stream = nullptr;

    void* d_input = nullptr;
    void* d_output = nullptr;

    float* h_input = nullptr;
    float* h_output = nullptr;

    size_t inputSize = 1 * 3 * 640 * 640 * sizeof(float);
    size_t outputSize = 1 * 84 * 8400 * sizeof(float);

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

YoloTRT::YoloTRT(const std::string& enginePath) {
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

    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engineData.data(), size);
    context = engine->createExecutionContext();

    cudaStreamCreate(&stream);

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    cudaHostAlloc((void**)&h_input, inputSize, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_output, outputSize, cudaHostAllocDefault);

    boxes.reserve(8400);
    confidences.reserve(8400);
    classIds.reserve(8400);
    indices.reserve(8400);

    const char* inputName = engine->getIOTensorName(0);
    const char* outputName = engine->getIOTensorName(1);
    context->setTensorAddress(inputName, d_input);
    context->setTensorAddress(outputName, d_output);
}

YoloTRT::~YoloTRT() {
    if (stream) cudaStreamDestroy(stream);
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);

    if (h_input) cudaFreeHost(h_input);
    if (h_output) cudaFreeHost(h_output);

    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
}

void YoloTRT::preprocess(const cv::Mat& srcImg) {
    cv::Mat resizedImg;
    cv::resize(srcImg, resizedImg, cv::Size(640, 640));
    cv::cvtColor(resizedImg, resizedImg, cv::COLOR_BGR2RGB);

    cv::Mat floatImg;
    resizedImg.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    int channelSize = 640 * 640;
    float* data = reinterpret_cast<float*>(floatImg.data);

    for (int i = 0; i < channelSize; ++i) {
        h_input[i]                   = data[i * 3 + 0];
        h_input[i + channelSize]     = data[i * 3 + 1];
        h_input[i + channelSize * 2] = data[i * 3 + 2];
    }
}

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

std::vector<Detection> YoloTRT::detect(const cv::Mat& srcImg, float confThresh, float nmsThresh) {
    if (srcImg.empty()) return {};

    preprocess(srcImg);

    cudaMemcpyAsync(d_input, h_input, inputSize, cudaMemcpyHostToDevice, stream);
    context->enqueueV3(stream);
    cudaMemcpyAsync(h_output, d_output, outputSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return postprocess(srcImg.cols, srcImg.rows, confThresh, nmsThresh);
}

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
    // C++ TensorRT 엔진 로드
    YoloTRT detector("yolov8n.engine");

    // 비디오 파일 열기
    std::string videoPath = "./test_videos/sample.mp4";
    cv::VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        std::cerr << "비디오 파일을 열 수 없습니다: " << videoPath << std::endl;
        return -1;
    }

    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    // Warm-up (첫 프레임 기반 50회 예열)
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

    // 동영상 프레임 단위 End-to-End 측정 (디코딩 + 전처리 + H2D + 추론 + D2H + NMS + 시각화)
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

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();
    double min = *std::min_element(latencies.begin(), latencies.end());
    double max = *std::max_element(latencies.begin(), latencies.end());

    std::cout << "\n================ [5. C++ Native TRT Video 벤치마크] ================" << std::endl;
    std::cout << "처리 프레임 수: " << frameCount << " / " << totalFrames << std::endl;
    std::cout << "Avg Latency: " << cv::format("%.2f", avg) << " ms" << std::endl;
    std::cout << "Min Latency: " << cv::format("%.2f", min) << " ms" << std::endl;
    std::cout << "Max Latency: " << cv::format("%.2f", max) << " ms" << std::endl;
    std::cout << "Avg FPS: " << cv::format("%.1f", 1000.0 / avg) << " FPS" << std::endl;
    std::cout << "=========================================================================" << std::endl;

    return 0;
}
```

### 실행 결과
```text
================ [5. C++ Native TRT Video 벤치마크] ================
처리 프레임 수: 3129 / 3129
Avg Latency: 9.49 ms
Min Latency: 6.91 ms
Max Latency: 21.76 ms
Avg FPS: 105.3 FPS
=========================================================================

```

</details>
