#include <drogon/drogon.h>
#include <iostream>

int main() {
    std::cout << "Starting Drogon C++ Server (WebSocket & REST API)..." << std::endl;

    // ⭐ Drogon이 받을 수 있는 최대 요청 크기를 500MB로 변경 (기본값: 1MB)
    drogon::app().setClientMaxBodySize(500 * 1024 * 1024);

    // 1. CORS 헤더 공통 추가
    drogon::app()
        .registerPostHandlingAdvice([](const drogon::HttpRequestPtr& req, const drogon::HttpResponsePtr& resp) {
            resp->addHeader("Access-Control-Allow-Origin", "*");
            resp->addHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            resp->addHeader("Access-Control-Allow-Headers", "*");
        });

    // 2. OPTIONS (Preflight) 요청 자동 통과 처리 (drogon_internal 제거)
    drogon::app().registerHandler(
        "/api/detect",
        [](const drogon::HttpRequestPtr& req, std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k200OK);
            resp->addHeader("Access-Control-Allow-Origin", "*");
            resp->addHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            resp->addHeader("Access-Control-Allow-Headers", "*");
            callback(resp);
        },
        {drogon::Options} // OPTIONS 메서드만 이 핸들러에서 받음
    );

    drogon::app()
        .addListener("0.0.0.0", 8889)
        .setThreadNum(8)
        .run();

    return 0;
}