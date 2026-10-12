#include "YoloTRT.h"

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