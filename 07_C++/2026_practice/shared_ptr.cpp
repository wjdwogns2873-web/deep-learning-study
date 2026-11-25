#include <iostream>
#include <memory>

struct Box {
    int value;
    Box() { std::cout << "Box 생성됨\n"; }
    ~Box() { std::cout << "Box 소멸됨\n"; }
};

int main() {
    std::cout << "--- main 시작 ---\n";
    {
        
    }
    return 0;
}