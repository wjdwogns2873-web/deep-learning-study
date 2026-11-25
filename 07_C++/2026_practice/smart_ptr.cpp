#include <iostream>

struct Box {
    int value;
    Box() { std::cout << "Box 생성됨\n"; }
    ~Box() { std::cout << "Box 소멸됨\n"; }
};

class MyPointer {
private:
    Box* ptr; // 진짜 힙 메모리 주소를 저장할 공간
public:
    MyPointer(Box* p) : ptr(p) {}

    // 복사 생성자 구현
    MyPointer(const MyPointer& other) {
        std::cout << "깊은 복사 발생! 힙에 새 Box를 만듭니다.\n";
        // other가 가리키는 Box의 값을 복사해서 새 Box를 만듬
        ptr = new Box(*other.ptr);
    }

    // 복사 대입 연산자 (copy assignment operator)
    MyPointer& operator=(const MyPointer& other) {
        std::cout << "대입 연산자 호출! (깊은 복사)\n";

        // 자기 자신을 대입하는지 확인
        if (this == &other) {
            std::cout << "삐빅.. 자기 자신이네용!\n";
            return *this; // 자기 자신이면 그냥 리턴
        }

        // 내가 원래 들고 있던 메모리는 낭비되니까 먼저 반납
        delete ptr;

        // 상대방의 Box 값을 복사해서 새 Box 만들기
        ptr = new Box(*other.ptr);

        return *this;
    }

    ~MyPointer() {
        std::cout << "MyPointer 소멸 (들고 있던 Box도 같이 delete)\n";
        delete ptr;
    }

    // 화살표 연산자 오버로딩 (ptr->value처럼 쓰기 위해)
    Box* operator->() const { return ptr; }

    // MyPointer(const MyPointer&) = delete; // 복사 금지
};

int main() {
    // std::cout << "--- main 시작 ---\n";
    // {
    //     MyPointer safeBox(new Box());
    //     safeBox->value = 42;

    //     MyPointer copyBox = safeBox;
    //     std::cout << "복사 완료. safeBox와 copyBox는 서로 다른 Box를 가리킵니다.\n";


    // } // 이 중괄호를 벗어나는 순간, safeBox(스택)가 사라지면서 소멸자가 호출됨



        MyPointer a(new Box());
        a->value = 10;

        MyPointer b(new Box());
        b->value = 99;

        std::cout << "--- 대입 연산 시작 --- \n";
        b = a; // 대입 연산자 호출
        std::cout << "b->value: " << b->value << "\n"; // 10이 나와야 함

    std::cout << "--- main 끝 ---\n";
    return 0;

    /*
    MyPointer* p = new MyPointer(new Box()); // 이렇게도 가능
    이렇게 하면 MyPointer도 힙에 생기지만 권장하지 않는다.
    MyPointer를 힙에 만들면 delete p를 또 기억해야 한다. 그래서 스마트 포인터는 항상 스택에 만드는 게 정석이다.
    결론: 스마트 포인터는 스택에 만들고, 관리할 객체만 힙에 만든다.
    */
}