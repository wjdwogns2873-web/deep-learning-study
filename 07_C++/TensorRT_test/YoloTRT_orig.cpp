#include "YoloTRT_orig.h"
namespace fs = std::filesystem;

YoloTRT::YoloTRT(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "엔진 파일을 찾을 수 없습니다: " << enginePath << std::endl;
        return; 
    }

    // 1. 맨 앞 4바이트를 읽어 Ultralytics 메타데이터 크기 확인
    uint32_t metaLen = 0;
    file.read(reinterpret_cast<char*>(&metaLen), sizeof(uint32_t));

    // 2. 전체 파일 크기 계산
    file.seekg(0, file.end);
    size_t totalSize = file.tellg();

    // 3. 메타데이터 길이만큼 건너뛴 위치(4 + metaLen)로 커서 이동
    size_t engineOffset = sizeof(uint32_t) + metaLen;
    size_t engineSize = totalSize - engineOffset;
    file.seekg(engineOffset, file.beg);

    // 4. 진짜 TensorRT 바이너리 영역만 읽어오기
    std::vector<char> engineData(engineSize);
    file.read(engineData.data(), engineSize);
    file.close();

    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engineData.data(), engineSize);
    context = engine->createExecutionContext();

    cudaStreamCreate(&stream);

    cudaMalloc(&d_input, maxInputSize);
    cudaMalloc(&d_output, maxOutputSize);

    cudaHostAlloc((void**)&h_input, maxInputSize, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_output, maxOutputSize, cudaHostAllocDefault);

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
    // 추론이 끝나면, 메모리 누수를 막기 위해 모든 자원을 반납하여 할당 해제합니다.
    if (stream) cudaStreamDestroy(stream);
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);

    if (h_input) cudaFreeHost(h_input);
    if (h_output) cudaFreeHost(h_output);

    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
}

void YoloTRT::preprocess(const cv::Mat& srcImg, int batchIdx) {
    cv::Mat resizedImg;
    cv::resize(srcImg, resizedImg, cv::Size(640, 640));
    cv::cvtColor(resizedImg, resizedImg, cv::COLOR_BGR2RGB);

    cv::Mat floatImg;
    resizedImg.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    int channelSize = 640 * 640;
    float* data = reinterpret_cast<float*>(floatImg.data);

    // batchIdx만큼 포인터 주소를 뒤로 밀어줍니다. 0번 이미지는 0에서 시작, 1번 이미지는 1,228,800(3 * 640 * 640)에서 시작..
    float* targetInput = h_input + (batchIdx * kSingleInputElements);

    for (int i = 0; i < channelSize; ++i) {
        targetInput[i]                   = data[i * 3 + 0]; // R
        targetInput[i + channelSize]     = data[i * 3 + 1]; // G
        targetInput[i + 2 * channelSize] = data[i * 3 + 2]; // B
    }
}

std::vector<Detection> YoloTRT::postprocess(int origW, int origH, float confThresh, float nmsThresh, int batchIdx) {
    boxes.clear();
    confidences.clear();
    classIds.clear();
    indices.clear();

    float scaleX = static_cast<float>(origW) / 640.0f;
    float scaleY = static_cast<float>(origH) / 640.0f;
    // batchIdx만큼 h_output의 포인터 주소를 뒤로 밀어줍니다. 0번 결과는 0에서 시작, 1번 결과는 705,600(84 * 8400)에서 시작..
    const float* targetOutput = h_output + (batchIdx * kSingleOutputElements);

    for (int i = 0; i < 8400; ++i) {
        float cx = targetOutput[0 * 8400 + i];
        float cy = targetOutput[1 * 8400 + i];
        float w  = targetOutput[2 * 8400 + i];
        float h  = targetOutput[3 * 8400 + i];

        float maxScore = 0.0f;
        int maxClassId = -1;
        for (int c = 0; c < 80; ++c) {
            float score = targetOutput[(c + 4) * 8400 + i];
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

            cv::Rect box(x1, y1, boxW, boxH);
            cv::Rect bounds(0, 0, origW, origH);
            box &= bounds;

            boxes.emplace_back(box);
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

    // 동적 배치 엔진에서도 안전하게 동작하도록 입력 상태를 1장으로 명시합니다.
    const char* inputName = engine->getIOTensorName(0);
    context->setInputShape(inputName, nvinfer1::Dims4{1, 3, 640, 640});

    preprocess(srcImg, 0);

    size_t singleInputBytes = kSingleInputElements * sizeof(float);
    size_t singleOutputBytes = kSingleOutputElements * sizeof(float);

    cudaMemcpyAsync(d_input, h_input, singleInputBytes, cudaMemcpyHostToDevice, stream);
    context->enqueueV3(stream);
    cudaMemcpyAsync(h_output, d_output, singleOutputBytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return postprocess(srcImg.cols, srcImg.rows, confThresh, nmsThresh, 0);
}

std::vector<std::vector<Detection>> YoloTRT::detectBatch(const std::vector<cv::Mat>& srcImgs, float confThresh, float nsmThresh) {
    int batchSize = static_cast<int>(srcImgs.size());
    if (batchSize == 0 || batchSize > kMaxBatchSize) return {};

    // 배치 사이즈만큼 입력 shape을 설정합니다. 꼭 4(maxBatchSize)가 아닐 수도 있습니다.
    const char* inputName = engine->getIOTensorName(0);
    context->setInputShape(inputName, nvinfer1::Dims4{batchSize, 3, 640, 640});

    // 배치에 들어온 이미지를 h_input 버퍼에 0번 부터 차례대로 전처리해서 채워넣습니다.
    for (int b = 0; b < batchSize; ++b) {
        preprocess(srcImgs[b], b);
    }

    // 실제 들어온 배치사이즈 만큼만 바이트 크기를 계산해서 복사합니다. cudaMemcpyAsync
    size_t currentInputBytes = batchSize * kSingleInputElements * sizeof(float);
    size_t currentOutputBytes = batchSize * kSingleOutputElements * sizeof(float);

    cudaMemcpyAsync(d_input, h_input, currentInputBytes, cudaMemcpyHostToDevice, stream);
    context->enqueueV3(stream);
    cudaMemcpyAsync(h_output, d_output, currentOutputBytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 이 함수의 반환형입니다. 하나의 프레임(이미지)은 std::vector<Detection>을 배출하는데, 배치 안에 여러장의 프레임이 있으므로
    // std::vector<Detection>을 std::vector로 감싸야 합니다.
    std::vector<std::vector<Detection>> batchResults;
    batchResults.reserve(batchSize); // 배치 사이즈만큼 메모리 공간을 예약합니다.

    // 배치의 각 이미지마다 후처리를 한 결과를 벡터(batchResults)에 담습니다.
    for (int b = 0; b < batchSize; ++b) {
        batchResults.push_back(postprocess(srcImgs[b].cols, srcImgs[b].rows, confThresh, nsmThresh, b));
    }

    return batchResults;
}

void YoloTRT::inferOnly() {
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
}