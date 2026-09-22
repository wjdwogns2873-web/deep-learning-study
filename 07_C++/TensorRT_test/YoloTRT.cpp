#include "YoloTRT.h"

YoloTRT::YoloTRT(const std::string& enginePath) {
    // 엔진 파일 열기
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

    // TensorRT 객체 생성
    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engineData.data(), size);
    context = engine->createExecutionContext();

    // CUDA Stream 생성
    cudaStreamCreate(&stream);

    // GPU VRAM 할당
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    // Host 버퍼 크기 확보
    inputHost.resize(1 * 3 * 640 * 640);
    outputHost.resize(1 * 84 * 8400);
}

YoloTRT::~YoloTRT() {
    // RAII: 객체가 파괴될 때 GPU/CPU 메모리를 자동으로 완전 해제
    if (stream) cudaStreamDestroy(stream);
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
}

void YoloTRT::preprocessLetterbox(const cv::Mat& srcImg, LetterboxInfo& info) {
    int imgW = srcImg.cols;
    int imgH = srcImg.rows;

    info.scale = std::min(640.0f / imgW, 640.0f / imgH);
    int newUnpadW = static_cast<int>(std::round(imgW * info.scale));
    int newUnpadH = static_cast<int>(std::round(imgH * info.scale));

    info.dw = (640 - newUnpadW) / 2;
    info.dh = (640 - newUnpadH) / 2;

    cv::Mat resizedImg;
    cv::resize(srcImg, resizedImg, cv::Size(newUnpadW, newUnpadH));

    int top = info.dh;
    int bottom = 640 - newUnpadH - info.dh;
    int left = info.dw;
    int right = 640 - newUnpadW - info.dw;

    cv::Mat letterboxImg;
    cv::copyMakeBorder(resizedImg, letterboxImg, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::cvtColor(letterboxImg, letterboxImg, cv::COLOR_BGR2RGB);
    letterboxImg.convertTo(letterboxImg, CV_32FC3, 1.0 / 255.0);

    int channelSize = 640 * 640;
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; i++) {
        channels[i] = cv::Mat(letterboxImg.cols, letterboxImg.rows, CV_32FC1, inputHost.data() + i * channelSize);
    }
    cv::split(letterboxImg, channels);
}

std::vector<Detection> YoloTRT::postprocess(const LetterboxInfo& info, int origW, int origH, float confThresh, float nmsThresh) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    // output Tensor: [1, 84, 8400]
    for (int i = 0; i < 8400 ; i++) {
        float cx = outputHost[0 * 8400 + i];
        float cy = outputHost[1 * 8400 + i];
        float w = outputHost[2 * 8400 + i];
        float h = outputHost[3 * 8400 + i];

        float maxScore = 0.0f;
        int maxClassId = -1;
        for (int c = 0; c < 80; c++) {
            float score = outputHost[(c + 4) * 8400 + i];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }

        if (maxScore > confThresh) {
            float x1 = ((cx - w / 2) - info.dw) / info.scale;
            float y1 = ((cy - h / 2) - info.dh) / info.scale;
            float boxW = w / info.scale;
            float boxH = h / info.scale;

            int rectX = std::max(0, std::min(static_cast<int>(x1), origW - 1));
            int rectY = std::max(0, std::min(static_cast<int>(y1), origH - 1));
            int rectW = std::max(0, std::min(static_cast<int>(boxW), origW - rectX));
            int rectH = std::max(0, std::min(static_cast<int>(boxH), origH - rectY));

            boxes.emplace_back(cv::Rect(rectX, rectY, rectW, rectH));
            confidences.emplace_back(maxScore);
            classIds.emplace_back(maxClassId);
        }
    }

    std::vector<int> indices;
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

    LetterboxInfo info;
    preprocessLetterbox(srcImg, info);

    // void* bindings[2] = { d_input, d_output };
    const char* inputName = engine->getIOTensorName(0); // 첫 번째 바인딩 (입력)
    const char* outputName = engine->getIOTensorName(1); // 두 번째 바인딩 (출력)

    context->setTensorAddress(inputName, d_input);
    context->setTensorAddress(outputName, d_output);

    // CPU -> GPU 복사
    // cudaMemcpy(d_input, inputHost.data(), inputSize, cudaMemcpyHostToDevice);
    // 비동기 H2D (Host -> Device) 메모리 복사
    cudaMemcpyAsync(d_input, inputHost.data(), inputSize, cudaMemcpyHostToDevice, stream);

    // context->executeV2(bindings);
    // 비동기 추론 실행 (enqueueV3 사용)
    context->enqueueV3(stream);

    // GPU -> CPU 복사
    // cudaMemcpy(outputHost.data(), d_output, outputSize, cudaMemcpyDeviceToHost);
    // 비동기 D2H (Device -> Host) 메모리 복사
    cudaMemcpyAsync(outputHost.data(), d_output, outputSize, cudaMemcpyDeviceToHost, stream);

    // 스트림 동기화 (결과를 읽기 직전 GPU 작업 완료 보장)
    cudaStreamSynchronize(stream);

    return postprocess(info, srcImg.cols, srcImg.rows, confThresh, nmsThresh);
}