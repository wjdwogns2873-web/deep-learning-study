#include <iostream>
#include <vector>
#include <memory> // 스마트 포인터 사용을 위한 헤더

int printVectorSize(const std::vector<float>& vec) {
    return vec.size();
}

// 100만개짜리 float vector를 쥐고 있는 스마트 포인터를 생성해서 반환하는 함수
std::unique_ptr<std::vector<float>> createBuffer() {
    // 100만 개의 float(0.0f로 초기화됨)을 갖는 vector를 스마트 포인터로 생성
    auto ptr = std::make_unique<std::vector<float>>(1000000);
    return ptr; // 포인터 자체를 반환
}

int main() {
    // createBuffer()가 반환한 스마트 포인터를 받음
    std::unique_ptr<std::vector<float>> myBuffer = createBuffer();

    // myBuffer는 포인터이므로 실제 vector 객체를 넘겨주기 위해 앞에 '*'를 붙여줍니다.
    int size = printVectorSize(*myBuffer);

    std::cout << "버퍼 크기: " << size << " 개" << std::endl;

    return 0; // main 함수가 끝나면서 myBuffer가 메모리에서 자동 해제됩니다.
}