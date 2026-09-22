#include "WebSocketController.h"

// Global 추론 엔진 인스턴스
static YoloORT g_yoloEngine("yolov8n.onnx", false);

void WebSocketController::handleNewMessage(const drogon::WebSocketConnectionPtr& wsConnPtr,
                                           std::string&& message,
                                           const drogon::WebSocketMessageType& type) {
    if (type == drogon::WebSocketMessageType::Binary) {
        std::vector<uchar> buffer(message.begin(), message.end());
        cv::Mat frame = cv::imdecode(buffer, cv::IMREAD_COLOR);

        if (frame.empty()) return;

        // C++ ONNX Runtime 추론
        auto detections = g_yoloEngine.detect(frame, 0.25f, 0.45f);

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
}

void WebSocketController::handleNewConnection(const drogon::HttpRequestPtr& req,
                                              const drogon::WebSocketConnectionPtr& wsConnPtr) {
    LOG_INFO << "새로운 프론트엔드 WebSocket 연결 성공!";
}

void WebSocketController::handleConnectionClosed(const drogon::WebSocketConnectionPtr& wsConnPtr) {
    LOG_INFO << "WebSocket 연결 종료";
}