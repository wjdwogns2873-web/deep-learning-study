#include <iostream>
#include <fstream>
#include <memory>
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
    // 1. Builder 생성
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    
    // 2. Network 생성 (최신 TRT에서는 NetworkDefinitionCreationFlag를 0으로 전달하거나 Explicit Batch가 기본값임)
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    
    // 3. ONNX Parser 생성
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser->parseFromFile("yolov8n.onnx", static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "ONNX 파싱 실패!" << std::endl;
        return -1;
    }

    // 4. Config 생성
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    // 5. C++ TensorRT 엔진 생성
    std::cout << "C++ TensorRT 엔진 생성 중..." << std::endl;
    nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);

    if (!plan) {
        std::cerr << "엔진 생성 실패!" << std::endl;
        return -1;
    }

    // 6. 엔진 파일 저장
    std::ofstream engineFile("yolov8n.engine", std::ios::binary);
    engineFile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    engineFile.close();

    std::cout << "성공적으로 호환되는 yolov8n.engine 이 생성되었습니다!" << std::endl;

    // 객체 정리
    delete plan;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return 0;
}