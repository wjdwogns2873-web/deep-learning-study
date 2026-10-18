# 9/24 보고서

## 모델 종류
1. PyTorch (.pt) 모델
2. FP32 ONNX Runtime 모델
3. PyCUDA / CuPy를 활용한 GPU 가속 모델

---

## 3번 방식(GPU 가속)의 핵심 장점

### (1) Host - Device 전송 오버헤드 파괴 (CuPy)
1번과 2번 방식은 이미지 한 장을 추론할 때마다 **[ CPU 전처리 ➔ GPU 전송 ➔ GPU 추론 ➔ CPU 수거 ]** 과정을 밟았다.
단건 탐질수록 이 오버헤드가 치명적이다.
반면 3번 방식은 이미지가 들어오자마자 GPU VRAM에 꽂아두고 내부에서 처리를 하기 때문에 오버헤드가 거의 없다.

### (2) 형변환 정규화 오버헤드 파괴
YOLO같은 모델은 HWC to CHW 그리고 float으로 정규화하는 전처리가 필수이다.
기존 방식(1, 2)은 CPU에서 NumPy로 수행하거나 추론 직전에 GPU에서 무겁게 변환했다.
3번 방식은, 이 전처리를 CUDA Stream을 사용해 GPU 내부에서 미리 끝내놓고 주소만 TensorRT로 보내기 때문에 추론 시점 오버헤드가 사라진다.

### (3) 추론 도중 메모리 할당(alloc) 오버헤드 제로
가장 결정적인 차이다. PyTorch나 ONNX Runtime은 메모리를 잡았다 풀었다 하는 최적화를 수행한다.
3번 방식은 `cuda.mem_alloc`을 통해 실행 전에 물리 메모리 주소를 고정해 두고, 그 통로만 재사용하므로 단건 다건 탐지에 상관없이 메모리 제어 오버헤드가 없어진다.

---

## 첫 번째 시도

### 1. PyTorch (.pt)

```python
image_path = 'apple_fuji_L_1-1.png'

# PyTorch (.pt)
_ = pt_model(image_path, verbose=False)
```
<details>
<summary><b>👇 전체 코드 보기</b></summary>

```python
image_path = 'apple_fuji_L_1-1.png'

# PyTorch (.pt)
_ = pt_model(image_path, verbose=False)

for i in range(5):
    # results[0]은 도대체 무엇인가?
    # results[0]은 배치 중에서 첫 번째 이미지를 의미합니다. 만약 배치가 2이고, 두 번째 이미지를 얻고싶다면 results[1]입니다.]
    # results[0].speed는 전처리(preprocess), 순수 추론(inference), 후처리(postprocess) 각 단계별로 소요시간을 딕셔너리로 담고 있습니다. ex: {'preprocess': 1.2ms, 'inference': 8.5ms, 'postprocess': 0.5ms}
    results = pt_model(image_path, verbose=False) 
    speed_dict = results[0].speed

    # 그래서 각 단계(전처리, 순수 추론, 후처리)의 걸린 모든 시간 합은 sum(~.value())로 표현합니다.
    total_time = sum(speed_dict.values()) 
    print(f'총 소요 시간: {total_time:.2f} ms')

    # 한가지 짚고 넘어가야 할 것. time.time()으로 잰 시간과 results[0].speed의 합이 다른 이유?
    # 1. 파이썬 time.time()
    # - 파이썬 레벨에서 함수가 호출되고 끝날 때까지의 시간을 잽니다.
    # - 파이썬 인터프리터 오버헤드, 객체 생성/할당 시간 등 부수적인 파이썬 레벨의 시간까지 모두 포함하기 때문에 speed의 합보다 더 소요됩니다.
    # 2. results[0].speed
    # - C++ 및 CUDA 레벨에서 GPU가 실제로 연산을 시작하고 끝낸 순수 GPU 처리 시간만 있고, 파이썬 오버헤드는 제외됩니다.
```
</details>

**실행 결과:**
```text
총 소요 시간: 10.27 ms
총 소요 시간: 10.83 ms
총 소요 시간: 9.99 ms
총 소요 시간: 10.12 ms
총 소요 시간: 9.93 ms
```

> **결론:** 파이썬 오버헤드 시간을 제외한 순수 추론 위주라서 공정한 방법은 아니었습니다.


### 2. ONNX (.onnx)

```python
onnx_path = '/content/best.onnx'

# ONNX 추론 세션 생성
# .onnx 모델 파일을 메모리에 올리고, CUDA(GPU)를 통해 연산하도록 provider를 지정하였습니다.
session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
```
<details>
<summary><b>👇 전체 코드 보기</b></summary>

```python
onnx_path = '/content/best.onnx'

# ONNX 추론 세션 생성
# .onnx 모델 파일을 메모리에 올리고, CUDA(GPU)를 통해 연산하도록 provider를 지정하였습니다.
session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])

# ONNX 모델은 내부적으로 입력/출력 노드의 이름(name)을 기반으로 데이터를 주고받기 때문에, 이름 추출이 필요합니다.
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

img = cv2.imread(image_path)

# (640, 640)으로 리사이즈하고, BGR을 RGB 순서로 바꿔줍니다.
# 그리고 원래 (640, 640, 3)의 텐서 순서를 transpose함으로써 (3, 640, 640)으로 변경해줍니다.
# 그리고 0과 1 사이의 실수 값으로 정규화도 해줍니다.
# 마지막으로, (3, 640, 640)을 모델 입력에 넣기 위해 배치 차원을 추가합니다.
img_resized = cv2.resize(img, (640, 640))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
input_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
input_tensor = np.expand_dims(input_tensor, axis=0)

# 첫 모델 추론시 GPU 드라이버 맵핑 및 메모리 할당 초기화가 발생하므로
# 본격적인 벤치마킹 측정 전에 미리 한번 실행하여 예열합니다. 웜업.
_ = session.run([output_name], {input_name: input_tensor})

for i in range(5): 
    start_time = time.time()
    img = cv2.imread(image_path) 
    img_resized = cv2.resize(img, (640, 640)) 
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) 
    input_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0 
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # 실제 추론 실행합니다.
    # 지정한 출력 이름(output_name)으로 결과 텐서를 받아오는 것입니다.
    # 입력은 딕셔너리 형태로 입력 이름(input_name)과 텐서 데이터(input_tensor)를 매핑하여 모델에게 전달합니다.
    outputs = session.run([output_name], {input_name: input_tensor}) 

    latency = (time.time() - start_time) * 1000 
    print(f'ONNX Runtime 전처리 & 추론 소요시간: {latency:.2f} ms')
```
</details>

**실행 결과:**
```text
ONNX Runtime 전처리 & 추론 소요시간: 127.06 ms
ONNX Runtime 전처리 & 추론 소요시간: 133.18 ms
ONNX Runtime 전처리 & 추론 소요시간: 124.63 ms
ONNX Runtime 전처리 & 추론 소요시간: 119.34 ms
ONNX Runtime 전처리 & 추론 소요시간: 130.24 ms
```

> 왜 이렇게 느려졌나요?
> 
> CPU 기반 OpenCV 전처리 때문입니다.
> 
> `cv.imread`는 하드디스크에서 매번 이미지를 읽어와서 디코딩하기 때문에 엄청 느립니다.
> 
> `cv.resize`와 `cvtColor`는 CPU가 행렬을 640 640으로 변환하고 채널을 바꾸는 연산입니다.
> 
> `input_tensor`를 255.0으로 나누는 게 가장 치명적인 CPU 병목 구간이라고 합니다.
> 
>  640x640x3 배열 전체 원소를 CPU가 일일이 255로 나누는 연산이 필요하기 때문에 상당히 오래 걸립니다.


### 3. TensorRT (.engine) - 메모리 제어 없음, 제로카피 안 함

```python
trt_model = YOLO(trt_path)

# Warm up
_ = trt_model(image_path, verbose=False)
```
<details>
<summary><b>👇 전체 코드 보기</b></summary>

```python
trt_model = YOLO(trt_path)

# Warm up
_ = trt_model(image_path, verbose=False)

for i in range(5): 
    start_time = time.time() 
    results = trt_model(image_path, verbose=False) 
    latency = (time.time() - start_time) * 1000 
    speed_dict = results[0].speed 
    print(f'Latency: {latency:.2f} | Speed: {sum(speed_dict.values()):.2f} ms')
```
</details>

**실행 결과:**
```text
Loading /content/best.engine for TensorRT inference...
Latency: 103.48 | Speed: 23.67 ms
Latency: 129.39 | Speed: 25.28 ms
Latency: 108.96 | Speed: 22.62 ms
Latency: 148.15 | Speed: 26.11 ms
Latency: 105.76 | Speed: 17.32 ms
```

> **해석:**
> 전체 Latency가 100ms에 달할 만큼 왜이리 느린가요?
>
> Ultralytics 함수(trt_model(image_path) 내부의 비효율적인 파이프라인 구조 때문입니다.
>
> 매 추론바다 CPU가 디스크에서 이미지 경로를 파싱하고, 전처리를 수행한 뒤, GPU로 매번 새로운 메모리를 동적으로 할당하고 복사하는 과정에서 심각한 병목이 발생했습니다.
>
> Speed를 보면 20ms대로 모델 자체 성능은 좋지만, 전처리 및 CPU가 GPU로 메모리를 전달하는 과정에서 병목이 걸려서 오래 걸렸습니다.


### 4. TensorRT (.engine) - 메모리 제어, 제로카피 사용

```python
# Logger란 무엇인가?
# 모델 빌드 중에 에러가 발생하면 그 에러를 수집하고 출력하는 역할을 합니다.
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
```
<details>
<summary><b>👇 전체 코드 보기</b></summary>

```python
# Logger란 무엇인가?
# 모델 빌드 중에 에러가 발생하면 그 에러를 수집하고 출력하는 역할을 합니다.
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Builder
# 말 그대로 빌더입니다. 이 모델을 GPU에 맞게 최적화된 상태로 구워 달라고 명령을 내리는 빌더입니다.
builder = trt.Builder(TRT_LOGGER)

# 모델의 레이어와 구조를 담을 빈 설계도입니다.
# 아직 아무것도 없는 상태이고, 어기에 ONNX 모델의 레이어와 구조를 채워넣을 겁니다.
network = builder.create_network()

# ONNX 파일을 TensorRT가 이해할 수 있도록 변환해서 설계도(network)에 그려주는 역할입니다.
parser = trt.OnnxParser(network, TRT_LOGGER)

# rb: read binary. 바이너리 형태로 ONNX 파일을 읽어낸 뒤에, 파싱(parse)합니다.
with open(onnx_path, 'rb') as model: 
    parser.parse(model.read())

# config는 이 모델을 굽는데 필요한 설정(최적화를 위한 옵션)입니다. 아무 설정도 하지 않을 수도 있습니다.
config = builder.create_builder_config()

# 이 설정(config)을 바탕으로 설계도(network)를 구워서 직렬화된 바이트 데이터로 만듭니다.
serialized_engine = builder.build_serialized_network(network, config)

# 읽은 걸 바탕으로 wb. write binary하여 파일을 써서(write), 엔진(.engine) 파일을 생성합니다.
clean_engine_path = '/content/clean.engine'
with open(clean_engine_path, 'wb') as f: 
    f.write(serialized_engine)

# 만들어진 engine 파일을 읽기 위해 Runtime 객체를 생성합니다.
runtime = trt.Runtime(TRT_LOGGER)

# 아까는 직렬화(build_serialized_network)했었지만, 실제 서비스를 할때는 비직렬화(deserialize_cuda_engine)해서 엔진 객체를 만듭니다.
with open(clean_engine_path, 'rb') as f: 
    engine = runtime.deserialize_cuda_engine(f.read())

# 엔진을 바탕으로 실제 추론을 실행할 context입니다.
context = engine.create_execution_context()

for i in range(5): 
    start_time = time.time() 
    src_img = cv2.imread(image_path) 

    # cp는 CuPy를 의미합니다. CPU에 있는 OpenCV 이미지 데이터를 GPU VRAM으로 복사해 올리는 함수입니다.
    gpu_img = cp.array(src_img)

    # 이 부분이 좀 어려운데, gpu_img는 (640, 640, 3)의 BGR 순서입니다.
    # 대괄호 안의 첫 콜론(:)은 맨 처음 640을 의미하고, 두번째 콜론(:)은 두 번째의 640을 의미합니다.
    # 즉, 640, 640을 다 가져오겠다는 뜻입니다.
    # 그리고 세번째가 ::-1인데, 이것은 처음부터 끝까지, 반대의 순서(-1)로 읽어라는 뜻입니다.
    # 지금 BGR로 되어있는 것을 RGB. 이렇게 반대로 해달라는 뜻입니다.
    gpu_img = gpu_img[:, :, ::-1]

    # zoom_factors: 원본 이미지를 640 640으로 맞추기 위한 비율입니다.
    # ndimage.zoom 함수에서 이 비율을 인자로 넣어서 640 640으로 리사이즈합니다.
    # order는 보간법(Interpolation)인데, 이미지를 늘리거나 줄일 때 빈 픽셀을 어떤 색으로 채울지 정하는 보간 방식입니다.
    # order=0이면 최근접 이웃법인데, 가장 단순하고 빠르지만 이미지가 깨질 수 있습니다.
    # order=1은 쌍선형 보간법인데, 픽셀과 픽셀 사이를 부드럽게 계산하여 리사이징하는데, 가장 많이 쓰이는 방식입니다.
    # order=2는 order=1에 비해 더 부드럽지만 연산량이 많아서, 지금 이 코드에서는 연산 속도와 화질의 균형을 위해서 order=1을 사용했습니다.
    zoom_factors = (640 / gpu_img.shape[0], 640 / gpu_img.shape[1], 1)
    gpu_resized = ndimage.zoom(gpu_img, zoom_factors, order=1)

    # (640, 640, 3)을 (3, 640, 640). 즉, HWC를 CHW로 텐서 순서를 변경합니다.    
    gpu_chw = cp.transpose(gpu_resized, (2, 0, 1))
    # 0과 1 사이의 실수 값으로 정규화합니다.
    gpu_input = gpu_chw.astype(cp.float32) / 255.0 

    output_name = engine.get_tensor_name(1) 
    output_shape = engine.get_tensor_shape(output_name)

    # 여기서 출력 텐서의 크기(itemsize)를 구하는 이유는 모델이 추론을 끝내고 뱉어낼 출력 텐서가 담길 공간을
    # GPU VRAM에 정확히 몇 바이트만큼 미리 확보해 놔야 하는지 계산하는 과정입니다.
    output_nbytes = trt.volume(output_shape) * np.dtype(np.float32).itemsize
    # 메모리를 할당하는 동시에, 그 할당된 메모리의 GPU 상 주소값을 반환해 줍니다.
    d_output = cuda.mem_alloc(output_nbytes) 

    # 모델의 입력과 출력 데이터가 실제로 저장될 메모리 주소를 TensorRT 엔진에게 직접 지정해 주는 작업입니다.
    context.set_tensor_address('images', int(gpu_input.data.ptr)) 
    context.set_tensor_address(output_name, int(d_output)) 

    # CUDA Stream: GPU에게 작업을 던지는 통로입니다.
    # 쓰는 이유?
    # 컴퓨터가 연산을 할 때 동기식으로 처리하면 [입력 복사 -> 완료될 때까지 대기 -> 추론 실행 -> 완료될 때까지 대기 -> 출력 복사] 순으로 멈추면서 일하게 됩니다.
    # 하지만 CUDA Stream을 만들어서 비동기 명령(execute_async_v3, memcpy_dtoh_async)을 던지면,
    # GPU가 연산을 하는 동안 CPU는 다음 프레임을 준비하거나 다른 작업을 동시에 수행할 수 있습니다.
    stream = cuda.Stream()

    context.execute_async_v3(stream_handle=stream.handle) 

    h_output = np.empty(output_shape, dtype=np.float32) 

    cuda.memcpy_dtoh_async(h_output, d_output, stream)

    # stream.synchronize()를 쓰는 이유
    # 지금까지 스트림에 던진 모든 GPU 작업이 끝날 때까지 대기해라. 동기화를 위해.
    stream.synchronize() 
    latency = (time.time() - start_time) * 1000 
    print(f'latency: {latency:.2f} ms')
```
</details>

**실행 결과:**
```text
latency: 28.39 ms
latency: 28.70 ms
latency: 28.68 ms
latency: 28.08 ms
latency: 28.49 ms
```

---

## 속도 측정에 대한 의문점 및 재측정 (공정한 비교)

### 1번 파이토치 모델(10ms대)이 4번 TensorRT 메모리 제어 제로카피 모델(28ms)보다 빨랐다 그 이유는?
원래는 반대여야 한다고 생각했음.

* 1번 모델의 `results[0].speed`는 파이썬 전체의 `time.time()`이 아님. 1번의 10ms는 이미지 디스크 읽기 시간이 없는 순수 연산 시간.
* 4번 모델은 하드디스크에서 이미지 바이너리를 읽어오는 `cv.imread` 시간이 포함되어 있다. 컴퓨터로 파일 읽는 건 상당히 오래 걸리는 작업이다.

실무 환경에서는 매번 `cv.imread`로 디스크에서 파일을 읽지 않는다. 카메라는 이미 메모리(RAM)에 프레임을 계속 올려두고 있기 때문에 `imread` 오버헤드가 없음.
공정하게 시간 측정하려면 **`cv.imread`(파일 읽기)를 제외한 순수 추론으로 비교해야 함.**

---

## 2차 재측정 결과 (순수 추론 비교)

### 1번 모델 순수 추론 재측정
```text
latency: 29.45 ms
latency: 13.80 ms
latency: 13.52 ms
latency: 13.45 ms
latency: 13.25 ms
```

### 4번 모델 순수 추론 재측정
```text
순수 TensorRT 가속 속도: 6.37 ms
순수 TensorRT 가속 속도: 6.05 ms
순수 TensorRT 가속 속도: 6.01 ms
순수 TensorRT 가속 속도: 6.02 ms
순수 TensorRT 가속 속도: 6.01 ms
```

---

## 🏆 최종 결과

**4번 모델 압승!**
