cmake_minimum_required(VERSION 3.10)
project(yolo_ort5)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Apple Silicon(M1/M2/M3/M4) Homebrew 경로 설정
if(APPLE)
    include_directories(/opt/homebrew/include)
    link_directories(/opt/homebrew/lib)
endif()

# OpenCV 설정
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime 경로 설정 (Homebrew 기본 경로 또는 사용자 지정 경로)
if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "/opt/homebrew")
endif()

include_directories("include/")
include_directories("${ONNXRUNTIME_DIR}/include")

add_executable(yolo_ort5
               main.cpp
               inference.cpp
               inference.h)

target_link_libraries(yolo_ort5
    ${OpenCV_LIBS} 
    "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib"
)

# ONNX Runtime 헤더 경로 명시
include_directories(/opt/homebrew/include)
include_directories(/opt/homebrew/include/onnxruntime)