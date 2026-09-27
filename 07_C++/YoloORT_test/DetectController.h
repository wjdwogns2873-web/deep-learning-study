#pragma once

#include <drogon/HttpController.h>
#include <opencv2/opencv.hpp>
#include "YoloORT.hpp"

class DetectController : public drogon::HttpController<DetectController> {
public:
    METHOD_LIST_BEGIN
    // POST /api/detect 요청을 detectImage 메소드에 매핑
    ADD_METHOD_TO(DetectController::detectImage, "/api/detect", drogon::Post);
    METHOD_LIST_END

    void detectImage(const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback);
};