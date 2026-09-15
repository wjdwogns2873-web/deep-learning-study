#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

void Detector(YOLO_V8*& p) {
    // 1. 현재 실행 경로 및 images 폴더 탐색
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images";

    // images 폴더가 없다면 현재 작업 디렉터리를 탐색하도록 대안 처리
    if (!std::filesystem::exists(imgs_path)) {
        imgs_path = current_path;
    }

    bool found_image = false;

    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            found_image = true;
            std::string img_path = i.path().string();
            std::cout << "\nProcessing image: " << img_path << std::endl;

            cv::Mat img = cv::imread(img_path);
            if (img.empty()) {
                std::cout << "Failed to load image: " << img_path << std::endl;
                continue;
            }

            std::vector<DL_RESULT> res;
            p->RunSession(img, res);

            std::cout << "Detected objects count: " << res.size() << std::endl;

            for (auto& re : res)
            {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                // Bounding Box 그리기
                cv::rectangle(img, re.box, color, 3);

                float confidence = std::floor(100.0f * re.confidence) / 100.0f;
                
                // 클래스 라벨 바운딩 처리 (예외 방지)
                std::string class_name = "Class " + std::to_string(re.classId);
                if (re.classId >= 0 && re.classId < static_cast<int>(p->classes.size())) {
                    class_name = p->classes[re.classId];
                }

                std::stringstream ss;
                ss << class_name << " " << std::fixed << std::setprecision(2) << confidence;
                std::string label = ss.str();

                int baseLine = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
                int top = std::max(re.box.y, labelSize.height);

                cv::rectangle(
                    img,
                    cv::Point(re.box.x, top - labelSize.height - 5),
                    cv::Point(re.box.x + labelSize.width, top + baseLine),
                    color,
                    cv::FILLED
                );

                cv::putText(
                    img,
                    label,
                    cv::Point(re.box.x, top - 2),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(255, 255, 255),
                    1
                );
            }

            cv::imshow("Result of Detection", img);
            std::cout << "Press ANY KEY on the image window to proceed..." << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }

    if (!found_image) {
        std::cout << "[Warning]: No image files (.jpg, .png, .jpeg) found in " << imgs_path << std::endl;
    }
}

int ReadCocoYaml(YOLO_V8*& p) {
    std::ifstream file("coco.yaml");
    if (!file.is_open())
    {
        std::cerr << "[Warning]: Failed to open coco.yaml file. Using default numeric labels." << std::endl;
        return 1;
    }

    std::string line;
    std::vector<std::string> names;
    bool in_names_section = false;

    while (std::getline(file, line))
    {
        if (line.find("names:") != std::string::npos) {
            in_names_section = true;
            continue;
        }

        if (in_names_section) {
            std::size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string name = line.substr(colon_pos + 1);
                // 공백, 따옴표, 쉼표 제거
                name.erase(0, name.find_first_not_of(" '\"["));
                name.erase(name.find_last_not_of(" '\"],") + 1);
                if (!name.empty()) {
                    names.push_back(name);
                }
            } else if (!line.empty() && line.find_first_not_of(" \t\r\n") != std::string::npos) {
                // 다른 섹션으로 넘어가면 종료
                break;
            }
        }
    }

    p->classes = names;
    std::cout << "Loaded " << p->classes.size() << " class names from coco.yaml" << std::endl;
    return 0;
}

void DetectTest()
{
    YOLO_V8* yoloDetector = new YOLO_V8;
    ReadCocoYaml(yoloDetector);
    
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.25f; // 신뢰도 임계값
    params.iouThreshold = 0.45f;
    params.modelPath = "yolov8n.onnx";
    params.imgSize = { 640, 640 };
    params.modelType = YOLO_DETECT_V8;
    params.cudaEnable = false;

    const char* err = yoloDetector->CreateSession(params);
    if (err != nullptr) {
        std::cerr << "Failed to create session: " << err << std::endl;
        delete yoloDetector;
        return;
    }

    Detector(yoloDetector);
    delete yoloDetector;
}

int main()
{
    // DetectTest 전용 실행
    DetectTest();
    return 0;
}