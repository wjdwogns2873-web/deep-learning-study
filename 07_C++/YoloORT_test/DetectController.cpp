#include "DetectController.h"
// #include <drogon/MultiPartParser.h>
#include <drogon/MultiPart.h>
#include <chrono>

static YoloORT fp32_yoloEngine("best.FP32.onnx", false);
static YoloORT fp16_yoloEngine("best.FP16.onnx", true);

void DetectController::detectImage(const drogon::HttpRequestPtr& req, 
                                   std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    // Multipart/form-data 파싱
    drogon::MultiPartParser fileParser;
    if (fileParser.parse(req) != 0) {
        Json::Value ret;
        ret["status"] = "error";
        ret["message"] = "Invalid multipart request";
        auto resp = drogon::HttpResponse::newHttpJsonResponse(ret);
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    // 폼 파라미터 추출 (conf, iou, model_type)
    auto parameters = fileParser.getParameters();
    float confThresh = parameters.count("conf_value") ? std::stof(parameters["conf_value"]) : 0.25f;
    float iouThresh = parameters.count("iou_value") ? std::stof(parameters["iou_value"]) : 0.45f;
    std::string modelType = parameters.count("model_type") ? parameters["model_type"] : "DROGON_CPP_FP16";

    auto files = fileParser.getFiles();
    if (files.empty()) {
        Json::Value ret;
        ret["status"] = "error";
        ret["message"] = "No image file provided";
        auto resp = drogon::HttpResponse::newHttpJsonResponse(ret);
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    auto& file = files[0];

    // 인메모리 바이너리를 cv::Mat으로 바로 디코딩 (I/O 오버헤드 최소화)
    std::vector<uchar> buffer(file.fileData(), file.fileData() + file.fileLength());
    cv::Mat frame = cv::imdecode(buffer, cv::IMREAD_COLOR);

    if (frame.empty()) {
        Json::Value ret;
        ret["status"] = "error";
        ret["message"] = "Failed to decode image";
        auto resp = drogon::HttpResponse::newHttpJsonResponse(ret);
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    // 추론 실행 및 소요 시간 측정
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Detection> detections;
    if (modelType == "DROGON_CPP_FP32") {
        detections = fp32_yoloEngine.detect(frame, confThresh, iouThresh);
    } else {
        detections = fp16_yoloEngine.detect(frame, confThresh, iouThresh);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double inferenceTimeMs = std::chrono::duration<double, std::milli>(end - start).count();

    // Bounding box 시각화
    for (const auto& det : detections) {
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = "Class " + std::to_string(det.classId) + ": " + cv::format("%.2f", det.confidence);
        cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    // 처리 완료된 이미지 JPEG 바이트로 인코딩
    std::vector<uchar> outputBuffer;
    cv::imencode(".jpg", frame, outputBuffer);

    // HTTP Response 반환
    std::string outputBlob(outputBuffer.begin(), outputBuffer.end());
    // auto resp = drogon::HttpResponse::newCustomHttpResponse(outputBlob, drogon::k200OK, drogon::CT_IMAGE_JPG);

    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setBody(outputBlob);
    resp->setStatusCode(drogon::k200OK);
    resp->setContentTypeCode(drogon::CT_IMAGE_JPG);

    // 프론트엔드 소요시간(ms) 표시용 헤더
    resp->addHeader("X-Inference-Time-MS", std::to_string(inferenceTimeMs));
    resp->addHeader("Access-Control-Expose-Headers", "X-Inference-Time-MS"); // CORS 브라우저 노출 설정

    callback(resp);
}