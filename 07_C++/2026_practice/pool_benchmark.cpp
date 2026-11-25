#include <iostream>
#include <vector>
#include <queue>
#include <memory>
#include <chrono>
#include <cstdlib>
#include <algorithm>

size_t g_allocCount = 0; // new가 호출된 총 횟수
size_t g_currentBytes = 0; // 현재 사용 중인 힙 메모리 크기
size_t g_peakBytes = 0; // 실행 중 찍었던 최대 힙 메모리

void* operator new(size_t size) {
    g_allocCount++;
    g_currentBytes += size;
    if (g_currentBytes > g_peakBytes) {
        g_peakBytes = g_currentBytes;
    }
    void* ptr = std::malloc(size);
    if (!ptr) throw std::bad_alloc();
    return ptr;
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, size_t size) noexcept {
    g_currentBytes -= size;
    std::free(ptr);
}

void resetMemoryStats() {
    g_allocCount = 0;
    g_currentBytes = 0;
    g_peakBytes = 0;
}

// 데이터 모델: 1장당 약 4.91MB(3 * 640 * 640 * 4 bytes) 크기의 가짜 이미지 패킷
struct ImageBuffer {
    static constexpr size_t kElements = 3 * 640 * 640;
    float* data; // 실제 거대 배열 포인터

    ImageBuffer() {
        data = new float[kElements]; // 생성될 때마다 4.91mb 힙 할당 발생
    }

    ~ImageBuffer() {
        delete[] data;
    }

    // 거대한 4.91mb 메모리가 실수로 복사되는 것을 방지
    ImageBuffer(const ImageBuffer&) = delete;
    ImageBuffer& operator=(const ImageBuffer&) = delete;
};

class SimpleObjectPool {
private:
    std::vector<std::unique_ptr<ImageBuffer>> storage_; // 실제 객체들의 소유권 보관소
    std::queue<ImageBuffer*> available_; // 현재 빌려줄 수 있는 객체 목록
public:
    // 생성자: 프로그램 시작시 poolSize만큼 미리 한 번만 할당해 둡니다.
    explicit SimpleObjectPool(size_t poolSize) {
        storage_.reserve(poolSize);
        for (size_t i = 0; i < poolSize; ++i) {
            auto buf = std::make_unique<ImageBuffer>();
            available_.push(buf.get()); // raw 포인터만 큐에 등록
            storage_.push_back(std::move(buf)); // 소유권은 storage_로 이동
        }
    }

    // 객체를 빌려주는 함수
    std::shared_ptr<ImageBuffer> acquire() {
        if (available_.empty()) {
            return nullptr;
        }

        ImageBuffer* rawPtr = available_.front();
        available_.pop();

        // 이 shared_ptr을 쓰는 곳이 모두 사라져 참조 카운트가 0이 되면, 
        // delete를 호출하는 대신 이 람다 함수가 실행되어 다시 available_ 큐에 반납됩니다.
        return std::shared_ptr<ImageBuffer>(rawPtr, [this](ImageBuffer* returndtr) {
            available_.push(returndtr);
        });
    }
};

int main() {
    const int kTotalFrames = 1000;

    volatile float dummySink = 0.0f; // 컴파일러가 루프 안의 연산을 통째로 날려버리지(최적화) 못하게 막는 장치

    // 실험 A: 오브젝트 풀 없이 매 프레임 동적 할당 (new and delete)
    resetMemoryStats();
    auto startA = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kTotalFrames; ++i) {
        // 매 프레임마다 4.91mb 생성 -> 작업 -> 스코프 벗어나며 소멸
        auto frame = std::make_shared<ImageBuffer>();

        // 1,228,800개의 모든 float 칸에 실제로 값을 채움
        std::fill(frame->data, frame->data + ImageBuffer::kElements, static_cast<float>(i));

        // 컴파일러에게 "이 배열의 마지막 값을 내가 진짜 쓰고 있다"라고 강제로 인식시킴
        dummySink = frame->data[ImageBuffer::kElements - 1];
        // frame->data[0] = static_cast<float>(i); // 메모리 쓰기 작업 시뮬레이션
    }

    auto endA = std::chrono::high_resolution_clock::now();
    double msA = std::chrono::duration<double, std::milli>(endA - startA).count();

    std::cout << "=== [실험 A] 매 프레임 동적 할당 (1000 프레임) ===" << std::endl;
    std::cout << "총 힙 할당(new) 횟수: " << g_allocCount << " 회" << std::endl;
    std::cout << "최대 힙 사용량(Peak): " << g_peakBytes / (1024.0 * 1024.0) << "MB" << std::endl;
    std::cout << "총 소요 시간: " << msA << "ms\n" << std::endl;

    // 실험 B: 오브젝트 풀 사용 (미리 4개 만들어두고 1000번 재사용)
    resetMemoryStats();
    SimpleObjectPool pool(4);
    size_t initAllocCount = g_allocCount; // 초기 세팅에 쓰인 할당 횟수 기록

    auto startB = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kTotalFrames; ++i) {
        // 풀에서 빌려옴 -> 작업 -> 스코프 벗어나면 커스텀 삭제자로 인해 풀에 자동 반납
        auto frame = pool.acquire();
        if (frame) {
            // 이미 확보된 4개의 메모리 방에 덮어쓰기만 실행
            std::fill(frame->data, frame->data + ImageBuffer::kElements, static_cast<float>(i));
            dummySink = frame->data[ImageBuffer::kElements - 1];

            // frame->data[0] = static_cast<float>(i);
        }
    }

    auto endB = std::chrono::high_resolution_clock::now();
    double msB = std::chrono::duration<double, std::milli>(endB - startB).count();

    std::cout << "=== [실험 B] 오브젝트 풀 재사용 (풀 크기 4개, 1000 프레임) ===" << std::endl;
    std::cout << "초기 풀 생성 할당 횟수: " << initAllocCount << " 회" << std::endl;
    std::cout << "1,000프레임 루프 중 추가 할당(new) 횟수: " << (g_allocCount - initAllocCount) << " 회" << std::endl;
    std::cout << "최대 힙 사용량(Peak): " << g_peakBytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "총 소요 시간: " << msB << " ms" << std::endl;

    return 0;

}