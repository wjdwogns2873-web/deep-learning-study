#include <iostream>
#include <vector>

// 1. 값 전달 (비효율적): vector 전체 데이터가 복사됨
void processBad(std::vector<int> vec) {
    // vec은 복사본임.
}

// 2. 참조 전달 (권장): 원본 메모리를 직접 가리킴
// const를 붙이면 "읽기 전용"으로 전달되어 데이터가 수정되는 것을 방지합니다.
void processGood(const std::vector<int>& vec) {
    std::cout << "크기: " << vec.size() << std::endl;
}

class ImageProcessor {
public:
    ImageProcessor() {std::cout << "이미지 프로세서 생성!" << std::endl;}
    ~ImageProcessor() {std::cout << "이미지 프로세서 소멸 (메모리 자동 해제)!" << std::endl; }

    void process() {std::cout << "이미지 처리 중..." << std::endl;}
};

int main() {
    {
        // std::make_unique<타입>() 으로 생성합니다.
        std::unique_ptr<ImageProcessor> processor = std::make_unique<ImageProcessor>();
        processor->process();
    } // <- 중괄호(스코프)를 벗어나는 순간, delete를 안 써도 소멸자(~ImageProcessor)가 자동 호출되어 메모리가 해제됩니다.

    std::cout << "main 함수 종료" << std::endl;
    return 0;
}