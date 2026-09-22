#include "WebSocketController.h"
#include <nlohmann/json.hpp>
#include <arpa/inet.h>


using json = nlohmann::json;

// Global 추론 엔진 인스턴스
static YoloORT fp32_yoloEngine("best.FP32.onnx", false);
static YoloORT fp16_yoloEngine("best.FP16.onnx", true);

void WebSocketController::handleNewMessage(const drogon::WebSocketConnectionPtr& wsConnPtr,
                                           std::string&& message,
                                           const drogon::WebSocketMessageType& type) {
    if (type != drogon::WebSocketMessageType::Binary) return;

    // 최소 패킷 크기 검증 (4바이트 헤더 포함 여부)
    if (message.size() < 4) return;

    // 앞 4바이트에서 JSON 데이터 길이(uint32_t) 읽기
    uint32_t jsonLen = 0;
    std::memcpy(&jsonLen, message.data(), sizeof(uint32_t));
    jsonLen = ntohl(jsonLen);

    // 패킷 유효성 검증
    if (message.size() < 4 + jsonLen) return;

    std::string jsonStr = message.substr(4, jsonLen);
    std::string modelType = "DROGON_CPP";

    try {
        auto j = json::parse(jsonStr);

        if (j.contains("model_type")) {
            modelType = j["model_type"].get<std::string>();
        }
    } catch (const std::exception& e) {
        // JSON 파싱 실패 시 예외 처리
    }

    // 핵심: JSON 끝점 이후부터 실제 이미지 바이너리 위치 오프셋 지정
    const char* imageBytes = message.data() + 4 + jsonLen;
    size_t imageSize = message.size() - 4 - jsonLen;

    // 이미지 디코딩
    std::vector<uchar> buffer(imageBytes, imageBytes + imageSize);
    cv::Mat frame = cv::imdecode(buffer, cv::IMREAD_COLOR);

    if (frame.empty()) return;

    // C++ ONNX Runtime 추론
    std::vector<Detection> detections;
    if (modelType == "DROGON_CPP_FP16") {
        detections = fp16_yoloEngine.detect(frame, 0.25f, 0.45f);
    } else if (modelType == "DROGON_CPP_FP32") {
        detections = fp32_yoloEngine.detect(frame, 0.25f, 0.45f);
    }

    // auto detections = g_yoloEngine.detect(frame, 0.25f, 0.45f);

    // Bounding Box 시각화
    for (const auto& det : detections) {
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = "Class " + std::to_string(det.classId) + ": " + cv::format("%.2f", det.confidence);
        cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    // 결과 프레임을 JPEG 바이너리 Blob으로 인코딩
    std::vector<uchar> outputBuffer;
    cv::imencode(".jpg", frame, outputBuffer);

    std::string outputBlob(outputBuffer.begin(), outputBuffer.end());
    wsConnPtr->send(outputBlob, drogon::WebSocketMessageType::Binary);
}

void WebSocketController::handleNewConnection(const drogon::HttpRequestPtr& req,
                                              const drogon::WebSocketConnectionPtr& wsConnPtr) {
    LOG_INFO << "새로운 프론트엔드 WebSocket 연결 성공!";
}

void WebSocketController::handleConnectionClosed(const drogon::WebSocketConnectionPtr& wsConnPtr) {
    LOG_INFO << "WebSocket 연결 종료";
}