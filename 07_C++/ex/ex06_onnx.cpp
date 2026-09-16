#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h> // ONNX Runtime C++ API 헤더

int main() {
    // ONNX Runtime 환경(Environment) 생성
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO_Inference");

    // 세션 옵션 설정
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1); // 스레드 수 설정

    std::cout << "ONNX Runtime C++ 환경 세팅 성공!" << std::endl;

    std::string modelPath = "best.FP32.onnx";
    Ort::Session session(env, modelPath.c_str(), sessionOptions);

    return 0;
}