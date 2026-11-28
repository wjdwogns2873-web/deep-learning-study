VS Code에서 작성한 TensorRT C++ 코드를 구글 코랩에서 아래와 같이 컴파일을 시도했다.

```python
!g++ -O3 -std=c++17 main.cpp Preprocessor.cpp YoloTRT.cpp Postprocessor.cpp TensorRTBuffer.cpp -o main \
    -I/usr/include/opencv4 \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    -lnvinfer -lcudart `pkg-config --cflags --libs opencv4`
```

그리고 엔진 파일(yolov8n.engine)은 아래와 같은 방식으로 export했다. (`!pip -q install ultralytics`는 필수)

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="engine")
```

그리고 컴파일 된 main 파일을 실행했더니 에러가 났다.
```python
!./main
```

```text
[TRT] IRuntime::deserializeCudaEngine: Error Code 1: Serialization (Serialization assertion header.magicTag == kEXPECTED_MAGIC_TAG failed.Trying to load an engine created with incompatible serialization version (1873 != 1953657958). Check that the engine was not created using safety runtime, same OS was used and version compatibility parameters were set accordingly and that it is a TRT engine file. In throwUnlessHeaderOk at /_src/runtime/dispatch/runtime.cpp:48)
terminate called after throwing an instance of 'std::runtime_error'
  what():  엔진 로드 실패
```

그 뒤로 원인을 분석하기 시작했고, 현재 구글 코랩에 깔린 TensorRT의 버전을 확인해 보았다. (`!pip -q install tensorrt`는 필수)
버전은 11.5.x가 나왔다.
```python
import tensorrt as trt
print("Python TRT:", trt.__version__)
```

```text
Python TRT: 11.5.0.63
```

아마 엔진 파일(yolov8n.engine)이 11.5 버전이 아니라서 에러가 나는 것으로 확인했다.

TensorRT 엔진 파일은 일종의 설명서인데, 이 설명서는 만들 때 쓴 TensorRT 버전에서만 읽을 수 있다.
그래서 아래와 같이 다시 엔진 파일을 구웠다.

```python
!trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine
```

아 그리고 처음엔 뭐가 문제인지 잘 몰라서 이것저것 시도하다가 컴파일할 때 `-L/usr/lib/x86_64-linux-gnu/` 이것도 추가해서 해결된 줄 알았는데, 

사실 코랩 환경에서는 기본 라이브러리 경로에 TensorRT가 이미 포함돼 있어서 -L 없이도 잘 되었다. 

진짜 문제는 엔진 파일 버전이었다.

그 뒤로 다시 main 파일을 실행했더니 성공적으로 빌드가 되었다.

---

나 혼자의 힘이 아니라 AI의 도움을 받아 원인을 분석하고, 내가 직접 trtexec로 엔진을 다시 구워서 해결했다.
