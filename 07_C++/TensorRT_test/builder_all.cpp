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