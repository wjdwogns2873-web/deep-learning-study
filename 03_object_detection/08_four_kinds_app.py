import io
import os
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO
import uuid
from datetime import datetime
import cv2

# 현재 시간을 '년월일시분초' 형식의 문자열로 추출 (20260720175830)
current_time_str = datetime.now().strftime("%Y%m%d%H%M%S")

# 고유 네임스페이스와 시간 문자열을 조합하여 UUID 생성
time_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, current_time_str)


app = FastAPI(title="Fruit Freshness Detection System")

model_path = "runs/four_kinds/weights/best.pt"
model = YOLO(model_path)

@app.get("/")
def read_root():
    return {"message": "Fruit Freshness Detection API is Running"}

@app.post("/predict")
async def predict_fruit(file: UploadFile = File(...)):
    # 파일 읽기 및 PIL 이미지로 변환
    file_bytes = await file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert('RGB')

    # YOLO 추론 진행
    results = model(image)
    
    print(f"results: \n{results}")
    # boxes: ultralytics.engine.results.Boxes object
    # keypoints: None
    # masks: None
    # names: {0: 'fresh_apple', 1: 'rotten_apple', 2: 'fresh_banana', 3: 'rotten_banana'}
    # obb: None
    # orig_img: array([[[...]]])
    # orig_shape: (344, 600)
    # path: 'image0.jpg'
    # probs: None
    # save_dir: '/opt/homebrew/runs/detect/predict'
    # semantic_mask: None
    # speed: {'preprocess': 1.1428749985498143, 'inference': 23.147709000113537, 'postprocess': 0.3489160008030012}]
    print(f"type of results: \n{type(results)}")
    # <class 'list'>
    result = results[0] # 첫 번째 이미지의 결과

    annotated_frame = result.plot()
    
    dest_dir = os.path.join(os.getcwd(), "dataset/integrated_dataset/FastAPI_results")
    wanted_filename = f"{current_time_str}.jpg"

    os.makedirs(dest_dir, exist_ok=True) # 폴더가 없으면 생성
    final_custom_path = os.path.join(dest_dir, wanted_filename)

    # OpenCV의 imwrite를 이용해 내가 원하는 이름으로 한 번에 저장
    cv2.imwrite(final_custom_path, annotated_frame)


    

    print(f"result.boxes: \n{result.boxes}")
    # ultralytics.engine.results.Boxes object with attributes:
    # cls: tensor([1., 1., 1., 1., 1., 1.])
    # tensor([0.9284, 0.9160, 0.9134, 0.8639, 0.8011, 0.6802])
    # data: tensor([[208.4596,  19.5697, 376.6910, 184.8582,   0.9284,   1.0000],
    #     [320.8998, 216.4702, 499.4646, 344.0000,   0.9160,   1.0000],
    #     [390.2737,   0.0000, 550.4515, 155.3632,   0.9134,   1.0000],
    #     [168.1662, 215.0695, 330.3897, 344.0000,   0.8639,   1.0000],
    #     [ 74.2605,  17.6333, 219.4610, 169.5220,   0.8011,   1.0000],
    #     [  1.3627,   0.0000, 129.6361, 127.1649,   0.6802,   1.0000]])
    # id: None
    # is_track: False
    # orig_shape: (344, 600)
    # shape: torch.Size([6, 6])
    # xywh: tensor([[292.5753, 102.2139, 168.2314, 165.2885],
    #     [410.1822, 280.2351, 178.5648, 127.5298],
    #     [470.3626,  77.6816, 160.1778, 155.3632],
    #     [249.2780, 279.5347, 162.2235, 128.9305],
    #     [146.8608,  93.5777, 145.2006, 151.8888],
    #     [ 65.4994,  63.5824, 128.2735, 127.1649]])
    # xywhn: tensor([[0.4876, 0.2971, 0.2804, 0.4805],
    #     [0.6836, 0.8146, 0.2976, 0.3707],
    #     [0.7839, 0.2258, 0.2670, 0.4516],
    #     [0.4155, 0.8126, 0.2704, 0.3748],
    #     [0.2448, 0.2720, 0.2420, 0.4415],
    #     [0.1092, 0.1848, 0.2138, 0.3697]])
    # xyxy: tensor([[208.4596,  19.5697, 376.6910, 184.8582],
    #     [320.8998, 216.4702, 499.4646, 344.0000],
    #     [390.2737,   0.0000, 550.4515, 155.3632],
    #     [168.1662, 215.0695, 330.3897, 344.0000],
    #     [ 74.2605,  17.6333, 219.4610, 169.5220],
    #     [  1.3627,   0.0000, 129.6361, 127.1649]])
    # xyxyn: tensor([[0.3474, 0.0569, 0.6278, 0.5374],
    #     [0.5348, 0.6293, 0.8324, 1.0000],
    #     [0.6505, 0.0000, 0.9174, 0.4516],
    #     [0.2803, 0.6252, 0.5506, 1.0000],
    #     [0.1238, 0.0513, 0.3658, 0.4928],
    #     [0.0023, 0.0000, 0.2161, 0.3697]])

    # 결과 데이터 정제
    detections = []
    for box in result.boxes:
        # 클래스 ID와 클래스 이름 매핑
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        confidence = float(box.conf[0].item())

        # 바운딩 박스 좌표 [xmin, ymin, xmax, ymax]
        bbox = [float(coord) for coord in box.xyxy[0].tolist()]

        detections.append({
            "class_name": class_name, 
            "class_id": class_id, 
            "confidence": round(confidence, 4), 
            "bbox": bbox
        })

    return {"counts": len(detections), "results": detections}