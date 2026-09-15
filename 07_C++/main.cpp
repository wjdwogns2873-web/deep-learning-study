#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "YOLOv8Detector.hpp"
#include "ThreadSafeQueue.hpp"

// 스레드간 안전한 제어를 위한 전역 원자적 변수
std::atomic<bool> isRunning(true);

// Producer 스레드: 동영상에서 프레임을 지속적으로 읽어 큐에 수집
void frameProducer(const std::string& videoPath, ThreadSafeQueue<cv::Mat>& inputQueue) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "영상 원본을 열 수 없습니다: " << videoPath << std::endl;
        isRunning = false;
        return;
    }

    // 동영상 원본의 FPS 가져오기
    double videoFps = cap.get(cv::CAP_PROP_FPS);
    if (videoFps <= 0) videoFps = 30.0;

    // 프레임당 지연 시간 계산 (예: 30FPS -> 약 33ms)
    int frameDelayMs = static_cast<int>(1000.0 / videoFps);

    cv::Mat frame;
    while (isRunning) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "영상 재생 완료 또는 프레임 수신 실패." << std::endl;
            isRunning = false;
            break;
        }

        inputQueue.push(frame.clone()); // 큐에 전달
        
        std::this_thread::sleep_for(std::chrono::milliseconds(frameDelayMs));
    }
}

// Consumer 스레드: 큐에서 최신 프레임을 꺼내 YOLOv8 추론 및 출력
void inferenceConsumer(YOLOv8Detector& detector, 
                       ThreadSafeQueue<cv::Mat>& inputQueue, 
                       ThreadSafeQueue<cv::Mat>& outputQueue) {
    cv::Mat frame;
    auto lastTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float fps = 0.0f;

    while (isRunning) {
        if (!inputQueue.pop(frame)) {
            continue;
        }

        // YOLOv8 추론
        std::vector<Detection> detections = detector.detect(frame);

        // FPS 계산
        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsedTime = currentTime - lastTime;
        if (elapsedTime.count() >= 1.0f) {
            fps = frameCount / elapsedTime.count();
            frameCount = 0;
            lastTime = currentTime;
        }

        // 검출된 결과를 현재 프레임 위에 그리기 (클래스 내부 메서드 또는 cv::rectangle)
        detector.drawResults(frame, detections, fps);

        outputQueue.push(frame);

        // cv::imshow("YOLOv8 Async Multi Thread Inference", frame); // 쓰면 안됩니다!!

        // if (cv::waitKey(1) == 27) { // ESC 키 누르면 종료 // 쓰면 안됩니다!!
        //     isRunning = false;
        //     break;
        // }
    }
}

int main() {
    try {
        // YOLOv8 객체 생성
        YOLOv8Detector detector("best.FP32.onnx");

        // 프레임 전달용 쓰레드 세이프 큐
        ThreadSafeQueue<cv::Mat> inputQueue(2);
        ThreadSafeQueue<cv::Mat> outputQueue(2);

        std::cout << "멀티스레드 실시간 추론 파이프라인을 시작합니다." << std::endl;

        std::string videoSource = "apple_video.mp4";

        // Producer & Consumer 스레드 생성 및 실행
        std::thread producer(frameProducer, videoSource, std::ref(inputQueue));
        std::thread consumer(inferenceConsumer, std::ref(detector), std::ref(inputQueue), std::ref(outputQueue));

        cv::Mat displayFrame;

        // 메인 스레드가 GUI 루프 가동
        while (isRunning) {
            if (outputQueue.pop(displayFrame)) {
                cv::imshow("YOLOv8 Async Multi Thread Inference", displayFrame);
            }

            if (cv::waitKey(1) == 27) {
                isRunning = false;
                break;
            }
        }

        // 두 스레드가 종료될 때까지 메인 스레드 대기
        if (producer.joinable()) producer.join();
        if (consumer.joinable()) consumer.join();

        cv::destroyAllWindows();
        std::cout << "파이프라인이 성공적으로 종료되었습니다." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "에러 발생: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

// int main() {
//     // 객체 생성 (생성자 호출 시 모델 자동 로드)
//     YOLOv8Detector detector("best.FP32.onnx");

//     // 비디오 캡처 준비
//     cv::VideoCapture cap("apple_video.mp4");
//     if (!cap.isOpened()) return -1;

//     cv::Mat frame;
//     while (true) {
//         cap >> frame;
//         if (frame.empty()) break;

//         // 단 한 줄로 객체 탐지 실행!
//         std::vector<Detection> results = detector.detect(frame);

//         detector.drawResults(frame, results);

//         cv::imshow("Class Modularized YOLOv8", frame);
//         if (cv::waitKey(1) == 'q') break;
//     }

//     cap.release();
//     cv::destroyAllWindows();
//     return 0;
// }