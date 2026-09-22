# 1. C++20 및 CMake 빌드 환경 준비 (Ubuntu 베이스)
FROM ubuntu:22.04 AS builder

# 백엔드 설치 중 프롬프트 대기 방지
ENV DEBIAN_FRONTEND=noninteractive

# 2. 필수 빌드 도구 및 C++ 라이브러리 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libjsoncpp-dev \
    uuid-dev \
    zlib1g-dev \
    libssl-dev \
    libsqlite3-dev \
    # OpenCV 설치
    libopencv-dev \
    # ONNX Runtime 빌드용
    wget \
    && rm -rf /var/lib/apt/lists/*

# 3. Drogon Framework 설치 (소스코드 빌드)
RUN git clone https://github.com/drogonframework/drogon.git /tmp/drogon && \
    cd /tmp/drogon && \
    git submodule update --init && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/drogon

# 4. 소스코드 복사 및 빌드
WORKDIR /app
COPY . .

RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)

# 5. 실행 명령 지정
EXPOSE 8888
CMD ["./build/drogon_app"]