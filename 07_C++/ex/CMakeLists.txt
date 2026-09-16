cmake_minimum_required(VERSION 3.10)
project(CppPractice)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 시스템에 설치된 OpenCV 패키지를 찾습니다.
find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    /opt/homebrew/include
    /opt/homebrew/include/onnxruntime
    /opt/homebrew/opt/onnxruntime/include
)

link_directories(
    /opt/homebrew/lib
    /opt/homebrew/opt/onnxruntime/lib
    )

# add_executable(ex01 ex01.cpp)
# add_executable(ex02 ex02.cpp)
# add_executable(ex03 ex03.cpp)
# add_executable(ex04_opencv ex04_opencv.cpp)
# add_executable(ex05_preprocess ex05_preprocess.cpp)
# add_executable(ex06_onnx ex06_onnx.cpp)
# add_executable(ex07_inference ex07_inference.cpp)
add_executable(ex08_video ex08_video.cpp)
add_executable(yolo_app main.cpp YOLOv8Detector.cpp)

# target_link_libraries(ex04_opencv PRIVATE ${OpenCV_LIBS})
# target_link_libraries(ex05_preprocess PRIVATE ${OpenCV_LIBS})
# target_link_libraries(ex06_onnx PRIVATE ${OpenCV_LIBS} onnxruntime)
# target_link_libraries(ex07_inference PRIVATE ${OpenCV_LIBS} onnxruntime)
target_link_libraries(ex08_video PRIVATE ${OpenCV_LIBS} onnxruntime)

target_link_libraries(yolo_app ${OpenCV_LIBS} onnxruntime)