#include <drogon/drogon.h>
#include <iostream>

int main() {
    std::cout << "Starting Drogon C++ WebSocket Server..." << std::endl;

    // Drogon 비동기 I/O 이벤트 루프 설정
    drogon::app()
        .registerPostHandlingAdvice([](const drogon::HttpRequestPtr& req, const drogon::HttpResponsePtr& resp) {
            resp->addHeader("Access-Control-Allow-Origin", "*");
            resp->addHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            resp->addHeader("Access-Control-Allow-Headers", "*");
        })
        .addListener("0.0.0.0", 8889)  // 8889 포트로 대기
        .setThreadNum(4)               // 워커 스레드 개수 설정
        .run();

    return 0;
}