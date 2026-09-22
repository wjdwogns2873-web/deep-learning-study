cmake_minimum_required(VERSION 3.10)
project(yolo_ort4 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Apple Silicon (M1~M4) Homebrew 기본 검색 경로 추가
if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
    include_directories(/opt/homebrew/include)
    link_directories(/opt/homebrew/lib)
endif()

# 2. OpenCV 설정
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 3. Drogon 웹 프레임워크 설정
find_package(Drogon REQUIRED)

# 4. ONNX Runtime 경로 설정 (Homebrew 경로)
if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "/opt/homebrew")
endif()

include_directories("include/")
include_directories("${ONNXRUNTIME_DIR}/include")
include_directories("${ONNXRUNTIME_DIR}/include/onnxruntime")

# 5. 소스 파일 및 실행 파일 지정
add_executable(yolo_ort4
    main.cpp
    YoloORT.cpp
    YoloORT.hpp
    WebSocketController.cpp
    WebSocketController.h
)

# 6. 라이브러리 링킹
target_link_libraries(yolo_ort4 PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib"
    Drogon::Drogon
)

# Apple 프레임워크 링킹 (Mac 전용)
if(APPLE)
    target_link_libraries(yolo_ort4 PRIVATE
        "-framework Foundation"
        "-framework CoreML"
    )
endif()