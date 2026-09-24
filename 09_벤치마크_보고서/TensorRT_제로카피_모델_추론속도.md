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
    results = pt_model(image_path, verbose=False) 
    speed_dict = results[0].speed 
    total_time = sum(speed_dict.values()) 
    print(f'총 소요 시간: {total_time:.2f} ms')
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

> **해석:** 순수 추론 위주임. 파이토치 내장 가속(CUDA 아키텍처) 덕분에 빠른 속도


### 2. ONNX (.onnx)

```python
onnx_path = '/content/best.onnx'

session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
```
<details>
<summary><b>👇 전체 코드 보기</b></summary>

```python
onnx_path = '/content/best.onnx'

session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
img = cv2.imread(image_path)
img_resized = cv2.resize(img, (640, 640))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
input_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
input_tensor = np.expand_dims(input_tensor, axis=0)

_ = session.run([output_name], {input_name: input_tensor})

for i in range(5): 
    start_time = time.time() 
    img = cv2.imread(image_path) 
    img_resized = cv2.resize(img, (640, 640)) 
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) 
    input_tensor = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0 
    input_tensor = np.expand_dims(input_tensor, axis=0) 
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

> **해석:**
> 느려진 원인은 루프 안에서 반복되는 CPU 기반 OpenCV 전처리 때문
> - `cv.imread`: 하드디스크에서 매번 이미지 바이너리를 읽어와 디코딩하는 속도 (엄청 느림)
> - `cv.resize` + `cvtColor`: CPU 연산 코어가 행렬을 640 640으로 쪼개고 채널을 바꾸는 연산
> - `input_tensor / 255.0`: 가장 치명적인 CPU 병목 구간. 640x640x3 배열 전체 원소를 CPU가 일일이 255로 나누기 연산하는 건 상당히 오래 걸림


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
> 스피드가 20ms 초반이란 것은 TensorRT 코어가 연산을 잘했다는 뜻임. 그럼에도 전체 시간(`time.time()`)이 100ms를 넘어버린 이유는..
> `trt_model(image_path)` 명령이 떨어질 때마다 내부적으로 이미지 경로 파싱하고 CPU가 이미지 전처리하고 PCIe를 거쳐 GPU로 매번 새 메모리를 할당하여 복사하느라 오버헤드 발생.


### 4. TensorRT (.engine) - 메모리 제어, 제로카피 사용

```python
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()
```
<details>
<summary><b>👇 전체 코드 보기</b></summary>

```python
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnx_path, 'rb') as model: 
    parser.parse(model.read())

config = builder.create_builder_config()
serialized_engine = builder.build_serialized_network(network, config)

clean_engine_path = '/content/clean.engine'
with open(clean_engine_path, 'wb') as f: 
    f.write(serialized_engine)

runtime = trt.Runtime(TRT_LOGGER)

with open(clean_engine_path, 'rb') as f: 
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

for i in range(5): 
    start_time = time.time() 
    src_img = cv2.imread(image_path) 
    
    gpu_img = cp.array(src_img) 
    gpu_img = gpu_img[:, :, ::-1] 

    zoom_factors = (640 / gpu_img.shape[0], 640 / gpu_img.shape[1], 1) 
    gpu_resized = ndimage.zoom(gpu_img, zoom_factors, order=1) 
    gpu_chw = cp.transpose(gpu_resized, (2, 0, 1)) 
    gpu_input = gpu_chw.astype(cp.float32) / 255.0 

    output_name = engine.get_tensor_name(1) 
    output_shape = engine.get_tensor_shape(output_name) 
    output_nbytes = trt.volume(output_shape) * np.dtype(np.float32).itemsize 
    d_output = cuda.mem_alloc(output_nbytes) 

    context.set_tensor_address('images', int(gpu_input.data.ptr)) 
    context.set_tensor_address(output_name, int(d_output)) 

    stream = cuda.Stream() 

    context.execute_async_v3(stream_handle=stream.handle) 

    h_output = np.empty(output_shape, dtype=np.float32) 

    cuda.memcpy_dtoh_async(h_output, d_output, stream) 
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

## 속도 측정에 대한 의문점 및 재측정 (공정 비교)

### ❓ 1번 파이토치 모델(10ms대)가 4번 TensorRT 메모리 제어 제로카피 모델(28ms)보다 빨랐다 그 이유는?
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
