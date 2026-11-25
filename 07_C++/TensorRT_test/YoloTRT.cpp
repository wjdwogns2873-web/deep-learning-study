#include "YoloTRT.h"
namespace fs = std::filesystem;

YoloTRT::YoloTRT(const std::string& enginePath) {
    // YoloTRT.cpp 생성자 내부 파일 읽기 부분 수정
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "엔진 파일을 찾을 수 없습니다: " << enginePath << std::endl;
        return; 
    }

    // 1. 맨 앞 4바이트를 읽어 Ultralytics 메타데이터 크기 확인
    // uint32_t metaLen = 0;
    // file.read(reinterpret_cast<char*>(&metaLen), sizeof(uint32_t));

    // 2. 전체 파일 크기 계산
    file.seekg(0, file.end);
    size_t totalSize = file.tellg();

    // 3. 메타데이터 길이만큼 건너뛴 위치(4 + metaLen)로 커서 이동
    // size_t engineOffset = sizeof(uint32_t) + metaLen;
    // size_t engineSize = totalSize - engineOffset;
    file.seekg(0, file.beg);

    // 4. 진짜 TensorRT 바이너리 영역만 읽어오기
    std::vector<char> engineData(totalSize);
    file.read(engineData.data(), totalSize);
    file.close();

    runtime = nvinfer1::createInferRuntime(gLogger);
    // engineData에는 .engine 파일의 바이너리가 1차원 벡터로 담겨 있는데, .data()를 써서 그 1차원 주소의 시작값ㅇ르 가져옵니다.
    // 주소 시작값부터 size만큼의 데이터를 역직렬화해서 엔진을 생성합니다.
    engine = runtime->deserializeCudaEngine(engineData.data(), totalSize);
    context = engine->createExecutionContext();

    // GPU에서 CPU로 데이터를 전송할 통로(Stream)를 생성합니다.
    cudaStreamCreate(&stream);

    // 최대 4장 분량의 메모리를 미리 한 번만 할당해둡니다.
    cudaMalloc(&d_input, maxInputSize);
    cudaMalloc(&d_output, maxOutputSize);

    // cudaHostAlloc(): CUDA에서 Pinned memory를 Host(CPU) 측에 할당하는 함수입니다.
    // 일반적으로 malloc()을 통해 할당한 메모리는 OS에 의해 디스크로 페이징(메모리스왑)이 될 수 있는 pageable memory입니다.
    // 반면 cudaHostAlloc()으로 할당된 메모리는 물리 RAM에 고정할(Pinned) 수 있으므로, GPU가 Host Memory에 빠르게 접근할 수 있습니다.
    // 여기서 &로 넘겨주는 이유도 위와 마찬가지로, 문신처럼 새기기 위함입니다.
    // (void**)는 CUDA 함수와 타입을 맞추기 위한 강제 형변환인데,
    // cudaHostAlloc은 사용자가 메모리에 float을 담을지, int를 담을지 알 수 없으므로 어떤 포인터 주소든 다 받을 수 있도록 강제 형변환합니다.
    cudaHostAlloc((void**)&h_input, maxInputSize, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_output, maxOutputSize, cudaHostAllocDefault);

    // 각 벡터 안에는 최대 8400개의 탐지된 네모박스의 데이터가 들어올 수 있으므로 미리 메모리 공간을 예약(reserve)합니다.
    boxes.reserve(8400);
    confidences.reserve(8400);
    classIds.reserve(8400);
    indices.reserve(8400);

    const char* inputName = engine->getIOTensorName(0);
    const char* outputName = engine->getIOTensorName(1);
    // 모델의 입력과 출력 데이터가 실제로 저장될 메모리 주소(d_input, d_output)를 매핑해 줍니다.
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

// batchIdx 번째 이미지가 들어갈 h_input의 시작 포인터를 계산해 기록
void YoloTRT::preprocess(const cv::Mat& srcImg, int batchIdx) {
    cv::Mat resizedImg;
    cv::resize(srcImg, resizedImg, cv::Size(640, 640)); // 640 640 크기로 리사이즈합니다.
    cv::cvtColor(resizedImg, resizedImg, cv::COLOR_BGR2RGB); // BGR을 RGB 순서로 바꿉니다.

    cv::Mat floatImg;
    resizedImg.convertTo(floatImg, CV_32FC3, 1.0 / 255.0); // 0~1의 실수값으로 정규화합니다.

    int channelSize = 640 * 640;
    // reinterpret_cast: 이 메모리 주소의 해석 방식을 강제로 바꿔라는 뜻입니다.
    // cv::Mat 객체(floatImg)의 내부 데이터의 실제 주소는 uchar* 형태입니다.
    // 이 메모리 주소를 float* 형식을 가진 포인터로 취급해서, 앞으로 실수 단위로 값을 꺼내 쓸 수 있도록 해달라고 타입을 변환합니다.
    float* data = reinterpret_cast<float*>(floatImg.data);



    // batchIdx 만큼 포인터 주소를 뒤로 밀어줌 (0번 이미지 -> 0, 1번 이미지 -> 1)
    float* targetInput = h_input + (batchIdx * kSingleInputElements);



    // [R0, G0, B0, R1, G1, B1, ...]를 [R0, R1, ..., G0, G1, ..., B0, B1, ...]로 바꿉니다.
    // 그러면 HWC가 CHW로 바뀝니다. (640, 640, 3) -> (3, 640, 640)
    for (int i = 0; i < channelSize; ++i) {
        targetInput[i]                   = data[i * 3 + 0]; // R
        targetInput[i + channelSize]     = data[i * 3 + 1]; // G
        targetInput[i + channelSize * 2] = data[i * 3 + 2]; // B
    }
}

// 후처리 함수입니다.
std::vector<Detection> YoloTRT::postprocess(int origW, int origH, float confThresh, float nmsThresh, int batchIdx) {
    // 이전 프레임 때 사용했던 네 개의 벡터들을 깔끔하게 비워주고, 재사용합니다.
    boxes.clear();
    confidences.clear();
    classIds.clear();
    indices.clear();

    // 전처리 과정에서 원본 이미지를 640 640 크기로 리사이즈하는 과정이 있었기에, 탐지된 네모박스도 640 640 이미지 기준입니다.
    // 이것을 원본 사이즈로 복원하는 과정이 필요한데,
    // 원본 이미지의 가로 세로 길이를 각각 640으로 나눈 뒤에 그 값을 scaleX, scaleY로 저장합니다.
    // 나중에 rectangle하기 전에 이값을 곱해주어 원본 크기로 복원하기 위함입니다.
    float scaleX = static_cast<float>(origW) / 640.0f;
    float scaleY = static_cast<float>(origH) / 640.0f;

    // batchIdx 만큼 출력 버퍼 시작 주소를 뒤로 밀어줌
    const float* targetOutput = h_output + (batchIdx * kSingleOutputElements);

    // 탐지된 8400개의 박스에 대해 반복문을 실행합니다.
    for (int i = 0; i < 8400; i++) {
        // h_output은 (1, 84, 8400) 데이터를 일렬로 가지고 있고, 앞 4개(0~3)는 cx, cy, bw, bh를 가지고 있습니다.
        float cx = targetOutput[0 * 8400 + i];
        float cy = targetOutput[1 * 8400 + i];
        float w  = targetOutput[2 * 8400 + i];
        float h  = targetOutput[3 * 8400 + i];

        // 각 80개의 클래스 중에서 확신도가 가장 높은 클래스 아이디와, 그 클래스의 확신도 값을 가져오는 로직입니다.
        float maxScore = 0.0f;
        int maxClassId = -1;
        // 뒤 80개(4~83)는 클래스 아이디를 의미합니다.
        for (int c = 0; c < 80; c++) {
            float score = targetOutput[(c + 4) * 8400 + i];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }

        // 가장 높은 확신도를 가진 클래스의 확신도가 conf 임계값을 넘었을 때의 박스(그리고 확신도, 클래스 아이디)만 벡터에 담습니다.
        if (maxScore > confThresh) {
            float x1 = (cx - w / 2.0f) * scaleX;
            float y1 = (cy - h / 2.0f) * scaleY;
            float boxW = w * scaleX;
            float boxH = h * scaleY;

            cv::Rect box(x1, y1, boxW, boxH); // 탐지된 네모박스입니다.
            cv::Rect bounds(0, 0, origW, origH); // 원본 이미지입니다.

            // 탐지된 네모박스 구했다면서? 근데 왜 원본 이미지와 교집합을 또 구하는데?
            // YOLOv8 모델이 탐지한 네모박스 좌표가 원본 이미지를 벗어났을 수도 있다.
            // 예를 들어, 원본 이미지가 640 640이라고 가정하고, 탐지된 물체는 x1 값이 480이고, boxW가 170이라고 했을 때,
            // xmin은 480, xmax는 650(480 + 170)이 되어버리면 640을 초과하게 된다.
            // 이렇게 되면 오류가 발생할 수 있기 때문에, 원본 이미지와의 교집합을 구함으로써 보정하는 과정이 필요하다.
            box &= bounds; // 원본 이미지와 탐지된 네모박스의 교집합을 구합니다.

            // 탐지된 네모박스, 그 네모박스의 클래스 아이디, 클래스에 대한 확신도를 각각의 벡터에 담습니다.
            boxes.emplace_back(box);
            confidences.emplace_back(maxScore);
            classIds.emplace_back(maxClassId);
        }
    }

    // 탐지된 박스(boxes)와 확신도(confidences)를 담은 벡터, 그리고 확신도 임계값(confThresh), NMS 임계값(nmsThresh)을 통해
    // 총 8400개의 박스 중에 임계값을 넘는 인덱스가 몇번인지 추출합니다.
    // indices에는 만족하는 인덱스 번호가 담깁니다. (예: 54, 74, 5344)
    cv::dnn::NMSBoxes(boxes, confidences, confThresh, nmsThresh, indices);

    // 추출된 인덱스 번호들(indices)만을 가지고 Detection 구조체를 담는 벡터(detections)에 하나씩 밀어넣습니다.
    std::vector<Detection> detections;
    detections.reserve(indices.size());
    for (int idx : indices) {
        // 클래스 생성자에서 reserve로 메모리 공간을 미리 예약해 두었기 때문에, 불필요한 메모리 재할당이 발생하지 않습니다.
        detections.push_back({ boxes[idx], confidences[idx], classIds[idx] });
    }

    // 최종적으로 벡터를 반환합니다.
    return detections;
}

// 실제 추론 함수입니다.
std::vector<Detection> YoloTRT::detect(const cv::Mat& srcImg, float confThresh, float nmsThresh) {
    if (srcImg.empty()) return {};

    // 동적 배치 엔진에서도 안전하게 동작하도록 입력 형태를 1장으로 명시
    const char* inputName = engine->getIOTensorName(0);
    context->setInputShape(inputName, nvinfer1::Dims4{1, 3, 640, 640});

    preprocess(srcImg, 0);

    size_t singleInputBytes = kSingleInputElements * sizeof(float);
    size_t singleOuptutBytes = kSingleOutputElements * sizeof(float);

    // h_input(Host, CPU)에는 이미지의 RGB 정보가 담겨 있고, 그것을 d_input(Device, GPU)로 inputSize만큼 카피합니다.
    cudaMemcpyAsync(d_input, h_input, singleInputBytes, cudaMemcpyHostToDevice, stream);
    // 모델 추론을 진행합니다.
    context->enqueueV3(stream);
    // 추론을 끝낸 출력 텐서.. d_output(Device, GPU)에는 (1, 84, 8400)의 데이터가 메모리 공간에 담겨 있고, 그것을 h_output(Host, CPU)로 outputSize만큼 카피합니다.
    cudaMemcpyAsync(h_output, d_output, singleOuptutBytes, cudaMemcpyDeviceToHost, stream);
    // 정확한 추론 시간을 측정하기 위해 처리가 끝날 때까지 기다려 줍니다.
    cudaStreamSynchronize(stream);

    // 후처리 과정을 진행합니다.
    return postprocess(srcImg.cols, srcImg.rows, confThresh, nmsThresh, 0);
}

std::vector<std::vector<Detection>> YoloTRT::detectBatch(const std::vector<cv::Mat>& srcImgs, float confThresh, float nmsThresh) {
    int batchSize = static_cast<int>(srcImgs.size());
    if (batchSize == 0 || batchSize > kMaxBatchSize) {
        return {};
    }

    // TensorRT 컨텍스트에 이번 배치에 들어온 실제 이미지 개수(1~4장)를 알려줍니다.
    const char* inputName = engine->getIOTensorName(0);
    context->setInputShape(inputName, nvinfer1::Dims4{batchSize, 3, 640, 640});

    for (int b = 0; b < batchSize; ++b) {
        preprocess(srcImgs[b], b);
    }

    // 실제 들어온 batchSize 분량만큼만 바이트 크기를 계산해 복사합니다.
    size_t currentInputBytes = batchSize * kSingleInputElements * sizeof(float);
    size_t currentOutputBytes = batchSize * kSingleOutputElements * sizeof(float);

    cudaMemcpyAsync(d_input, h_input, currentInputBytes, cudaMemcpyHostToDevice, stream);
    context->enqueueV3(stream);
    cudaMemcpyAsync(h_output, d_output, currentOutputBytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<std::vector<Detection>> batchResults;
    batchResults.reserve(batchSize);

    for (int b = 0; b < batchSize; ++b) {
        batchResults.push_back( postprocess(srcImgs[b].cols, srcImgs[b].rows, confThresh, nmsThresh, b) );
    }

    return batchResults;
}

void YoloTRT::inferOnly() {
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
}

// 카메라 스레드에서 각자 병렬로 실행할 전처리 함수
void YoloTRT::preprocessSingle(const cv::Mat& srcImg, std::vector<float>& outBuffer) {
    if (outBuffer.size() != kSingleInputElements) {
        outBuffer.resize(kSingleInputElements);
    }

    cv::Mat resizedImg;
    cv::resize(srcImg, resizedImg, cv::Size(640, 640));
    cv::cvtColor(resizedImg, resizedImg, cv::COLOR_BGR2RGB);

    cv::Mat floatImg;
    resizedImg.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    int channelSize = 640 * 640;
    const float* data = reinterpret_cast<const float*>(floatImg.data);
    float* target = outBuffer.data();

    for (int i = 0; i < channelSize; ++i) {
        target[i]                   = data[i * 3 + 0];
        target[i + channelSize]     = data[i * 3 + 1];
        target[i + channelSize * 2] = data[i * 3 + 2];
    }
}

std::vector<std::vector<Detection>> YoloTRT::detectBatchPreprocessed(
    const std::vector<std::vector<float>>& batchBuffers, 
    const std::vector<cv::Size>& origSizes, 
    float confThresh, 
    float nmsThresh
) {
    int batchSize = static_cast<int>(batchBuffers.size());
    if (batchSize == 0 || batchSize > kMaxBatchSize) {
        return {};
    }

    const char* inputName = engine->getIOTensorName(0);
    context->setInputShape(inputName, nvinfer1::Dims4{batchSize, 3, 640, 640});

    size_t singleBytes = kSingleInputElements * sizeof(float);
    for (int b = 0; b < batchSize; ++b) {
        float* destPtr = h_input + (b * kSingleInputElements);
        std::memcpy(destPtr, batchBuffers[b].data(), singleBytes);
    }

    size_t currentInputBytes = batchSize * singleBytes;
    size_t currentOutputBytes = batchSize * kSingleOutputElements * sizeof(float);

    cudaMemcpyAsync(d_input, h_input, currentInputBytes, cudaMemcpyHostToDevice, stream);
    context->enqueueV3(stream);
    cudaMemcpyAsync(h_output, d_output, currentOutputBytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<std::vector<Detection>> batchResults;
    batchResults.reserve(batchSize);

    for (int b = 0; b < batchSize; ++b) {
        batchResults.push_back(postprocess(origSizes[b].width, origSizes[b].height, confThresh, nmsThresh, b));
    }

    return batchResults;
}