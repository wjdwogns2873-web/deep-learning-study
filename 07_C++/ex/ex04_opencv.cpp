#include <iostream>
#include <opencv2/opencv.hpp> // OpenCV C++ 핵심 헤더

int main() {
    // 100 100 크기의 3채널 이미지를 0(검은색)으로 생성
    // 파이썬의 np.zeros((100, 100, 3), dtype=np.uint8)
    cv::Mat image = cv::Mat::zeros(101, 100, CV_8UC3);

    // 이미지의 가로, 세로, 채널 수 출력
    std::cout << "가로(Width): " << image.cols << std::endl; // 100
    std::cout << "세로(Height): " << image.rows << std::endl; // 101
    std::cout << "채널(Channels): " << image.channels() << std::endl;

    // 이미지 리사이즈
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(640, 640));

    std::cout << "리사이즈 후 가로: " << resizedImage.cols << std::endl;
    std::cout << "리사이즈 후 세로: " << resizedImage.rows << std::endl;

    return 0;
}