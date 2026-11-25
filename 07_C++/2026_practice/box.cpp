#include <iostream>

struct Box {
    int value;
};

int main() {
    Box a;
    a.value = 10;
    std::cout << "a.value: " << a.value << "\n";

    Box* b = new Box;
    b->value = 20;
    std::cout << "b->value: " << b->value << "\n";

    // delete b;
    delete b;

    return 0;
}