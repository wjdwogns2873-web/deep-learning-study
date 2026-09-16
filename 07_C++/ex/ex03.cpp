#include <iostream>
#include <vector>
#include <memory>

void printImage(const std::vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); i++)
    {
        std::cout << vec[i] << std::endl;
    }
    
}

int main() {
    std::vector<float> img = {1.0f, 2.0f, 3.0f};
    printImage(img);

    return 0;
}