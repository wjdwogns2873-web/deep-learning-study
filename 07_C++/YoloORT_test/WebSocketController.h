#pragma once

#include <drogon/WebSocketController.h>
#include <opencv2/opencv.hpp>
#include "YoloORT.hpp"

class WebSocketController : public drogon::WebSocketController<WebSocketController> {
public:
    void handleNewMessage(const drogon::WebSocketConnectionPtr& wsConnPtr,
                          std::string&& message,
                          const drogon::WebSocketMessageType& type) override;

    void handleNewConnection(const drogon::HttpRequestPtr& req,
                             const drogon::WebSocketConnectionPtr& wsConnPtr) override;

    void handleConnectionClosed(const drogon::WebSocketConnectionPtr& wsConnPtr) override;

    WS_PATH_LIST_BEGIN
    WS_PATH_ADD("/yolo");
    WS_PATH_LIST_END
};