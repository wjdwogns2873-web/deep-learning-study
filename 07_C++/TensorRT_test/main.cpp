#include "YoloTRT.h"
#include <chrono>

// 비디오 스트리밍 탐지
int main() {
    YoloTRT detector("yolov8n.engine");

    cv::VideoCapture cap("sample_video.mp4");
    if (!cap.isOpened()) {
        std::cerr << "비디오 파일을 열 수 없습니다." << std::endl;
        return -1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // 결과 비디오 저장용 Writer
    cv::VideoWriter writer("output_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

    cv::Mat frame;
    int frameCount = 0;

    std::cout << "비디오 추론 시작.." << std::endl;

    while (cap.read(frame)) {
        auto start = std::chrono::high_resolution_clock::now();

        auto detections = detector.detect(frame, 0.25, 0.45);

        auto end = std::chrono::high_resolution_clock::now();
        double durationMs = std::chrono::duration<double, std::milli>(end - start).count();
        double currentFps = 1000.0 / durationMs;

        for (const auto& det : detections) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
            std::string label = "Class " + std::to_string(det.classId) + ": " + cv::format("%.2f", det.conf);
            cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        std::string fpsText = cv::format("FPS: %.1f (%.1f ms)", currentFps, durationMs);
        cv::putText(frame, fpsText, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        writer.write(frame);

        frameCount++;
        if (frameCount % 30 == 0) {
            std::cout << frameCount << " 프레임 처리 완료.. (" << fpsText << ")" << std::endl;
        }
    }

    std::cout << "모든 비디오 프레임 처리 완료! (output_video.mp4 저장)" << std::endl;
    return 0;
}

// 단건 탐지
// int main() {
//     // 객체 생성 (엔진 파일 로드 및 GPU 할당 자동 진행)
//     YoloTRT detector("yolov8n.engine");

//     cv::Mat img = cv::imread("bus.jpg");
//     if (img.empty()) {
//         std::cerr << "이미지 로드 실패" << std::endl;
//         return -1;
//     }

//     auto detections = detector.detect(img, 0.25f, 0.45f);

//     std::cout << "[OOP YoloTRT] 검출 객체 수: " << detections.size() << "개" << std::endl;

//     for (const auto& det : detections) {
//         cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), 2);
//         std::string label = "Class " + std::to_string(det.classId) + ": " + cv::format("%.2f", det.conf);
//         cv::putText(img, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
//     }

//     cv::imwrite("output_oop.jpg", img);
//     std::cout << "결과가 output_oop.jpg로 저장되었습니다." << std::endl;

//     return 0;
// }