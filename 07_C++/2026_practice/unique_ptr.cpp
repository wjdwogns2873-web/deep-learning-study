#include <iostream>
#include <memory>

struct Box
{
    int value;
    Box() { std::cout << "Box 생성됨\n"; }
    ~Box() { std::cout << "Box 소멸됨\n"; }
};

int main() {
    std::cout << "--- main 시작 ---\n";
    {
        // std::unique_ptr<Box> safeBox(new Box());
        // std::unique_ptr<Box> safeBox = std::make_unique<Box>(); // 이게 더 안전.

        auto p1 = std::make_unique<Box>();
        p1->value = 42;

        std::cout << "이동 전: p1이 비어있나? " << (p1 == nullptr ? "Yes" : "No") << "\n";

        // 소유권 이동
        auto p2 = std::move(p1);

        std::cout << "이동 후: p1이 비어있나? " << (p1 == nullptr ? "Yes" : "No") << "\n";
        std::cout << "이동 후 p2의 값: " << p2->value << "\n";
    } // 중괄호 벗어나면 소멸자가 자동 호출되고, delete도 자동
    std::cout << "--- main 끝 ---\n";
    return 0;
}
