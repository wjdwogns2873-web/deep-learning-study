#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat inputImage = cv::Mat::zeros(640, 640, CV_8UC3);

    cv::Mat rgbImage;
    cv::cvtColor(inputImage, rgbImage, cv::COLOR_BGR2RGB);

    // 0~255 uint8 값을 0.0~1.0 float32로 정규화
    cv::Mat floatImage;
    rgbImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

    int height = floatImage.rows;
    int width = floatImage.cols;
    int channels = floatImage.channels();

    // 1 * 3 * 640 * 640 크기의 1차원 C++ float 버퍼 준비
    std::vector<float> inputTensorValues(1 * channels * height * width);

    // OpenCV의 cv::split을 활요해 R, G, B 채널별로 메모리를 분리
    std::vector<cv::Mat> chwChannels(channels);
    for (int i = 0; i < channels; ++i) {
        std::cout << inputTensorValues.data() << std::endl;
        chwChannels[i] = cv::Mat(height, width, CV_32FC1, inputTensorValues.data() + i * height * width);
    }

    // floatImage 데이터를 chwChannels(R, G, B 분리된 버퍼)로 복사
    cv::split(floatImage, chwChannels);

    std::cout << "전처리 완! ONNX 입력을 위한 Tensor 버퍼 전체 크기: "
              << inputTensorValues.size() << " 개 (640*640*3 = " << 640*640*3 << ")" << std::endl;
    
    return 0;
}