#include "engine.hpp"

YOLOv8Detector::YOLOv8Detector(const std::string& modelPath, 
                               const std::string& logid, 
                               const std::string& provider)
    : env_(ORT_LOGGING_LEVEL_WARNING, logid.c_str()) {
    
    Ort::SessionOptions sessionOptions;
    if (provider == "CUDA") {
        OrtCUDAProviderOptions cudaOption;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions);

    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStringPtrs;
    Ort::AllocatorWithDefaultOptions inputNodeNamesAllocator;
    auto inputNodesCount = session_.GetInputCount();
    inputNames_.reserve(inputNodesCount);
    inputNamesCStr_.reserve(inputNodesCount);

    for (int i = 0; i < inputNodesCount; i++) {
        auto inputNodeNamePtr = session_.GetInputNameAllocated(i, inputNodeNamesAllocator);
        inputNames_.emplace_back( inputNodeNamePtr.get() );
        inputNamesCStr_.push_back( inputNames_.back().c_str() );
    }

    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStringPtrs;
    Ort::AllocatorWithDefaultOptions outputNodeNamesAllocator;
    auto outputNodesCount = session_.GetOutputCount();
    for (int i = 0; i < outputNodesCount; i++) {
        auto outputNodeNamePtr = session_.GetOutputNameAllocated(i, outputNodeNamesAllocator);
        outputNodeNameAllocatedStringPtrs.push_back( std::move(outputNodeNamePtr) );
        outputNames_.push_back( outputNodeNameAllocatedStringPtrs.back().get() );
    }

    // Convert output names to c_str
    for (const std::string& name : outputNames_) {
        outputNamesCStr_.push_back(name.c_str());
    }

    modelMetadata_ = session_.GetModelMetadata();

    Ort::AllocatorWithDefaultOptions metadataAllocator;
    std::vector<Ort::AllocatedStringPtr> metadataAllocatedKeys = modelMetadata_.GetCustomMetadataMapKeysAllocated(metadataAllocator);
    std::vector<std::string> metadataKeys;
    metadataKeys.reserve(metadataAllocatedKeys.size());

    for (const Ort::AllocatedStringPtr& allocatedStringPtr : metadataAllocatedKeys) {
        metadataKeys.emplace_back(allocatedStringPtr.get());
    }

    // Parse metadata
    for (const std::string& key : metadataKeys) {
        Ort::AllocatedStringPtr metadataValue = modelMetadata_.LookupCustomMetadataMapAllocated(key.c_str(), metadataAllocator);
        if (metadataValue != nullptr) {
            auto raw_metadata_value = metadataValue.get();
            metadataMap_[key] = std::string(raw_metadata_value);
        }
    }

    // Find the input size of the model
    auto imgsz_item = metadataMap_.find("imgsz");
    if (imgsz_item != metadataMap_.end()) {
        std::string copiedItem = imgsz_item->second;// 깊은 복사("[640, 640]")
        std::stringstream ss(copiedItem);
        char trash; // [, ], ,(쉼표) 문자를 건너뛰기 위한 변수
        int width, height;
        ss >> trash >> height >> trash >> width;
        // std::cout << "width: " << width << ", height: " << height << std::endl;
        cvSize_ = cv::Size(width, height);
    } else {
        cvSize_ = cv::Size(640, 640);
    }

    auto stride_item = metadataMap_.find("stride");
    if (stride_item != metadataMap_.end()) {
        int stride = std::stoi(stride_item->second);
        if (stride_ == -1) {
            stride_ = stride;
        }
    }

    // For the names of the classes
    auto names_item = metadataMap_.find("names");
    if (names_item != metadataMap_.end()) {
        std::unordered_map<int, std::string> names = parseNames(names_item->second);
        std::cout << "***Names from metadata***" << std::endl;
        for (const auto& pair : names) {
            std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
        }
        if (names_.empty()) {
            names_ = names;
        }
    }

    auto task_item = metadataMap_.find("task");
    if (task_item != metadataMap_.end()) {
        // std::string task
        if (task_.empty()) {
            task_ = task_item->second;
            std::cout << "task: " << task_ << std::endl;
        }
    }

    if (nc_ == -1 && names_.size() > 0) {
        nc_ = names_.size();
    }

    // cvSize_는 무조건 있음.if-else 구문을 거쳐서. 640(width), 640(height)
    if (inputTensorShape_.empty()) {
        inputTensorShape_ = { 1, 3,  cvSize_.height, cvSize_.width };
    }

    std::cout << "inputTensorShape: [";
    for (size_t i = 0; i < inputTensorShape_.size(); ++i) {
        std::cout << inputTensorShape_[i];
        if (i < inputTensorShape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
}

YOLOv8Detector::~YOLOv8Detector() {};

std::vector<Ort::Value> YOLOv8Detector::preprocess(cv::Mat& frame) {
    orgImgSize_ = frame.size();

    cv::Mat rgbFrame;
    cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB); // Convert to the RGB

    const bool auto_ = false;
    const bool scalefill_ = false;
    cv::Mat letterboxImage = letterbox(rgbFrame, cv::Scalar(), auto_, scalefill_, true, stride_);

    std::vector<float> blob = imageToBlob(letterboxImage); // HWC -> CHW

    int64_t inputTensorSize = vector_product(inputTensorShape_);

    inputTensorValues_.resize(inputTensorSize);
    std::copy(blob.begin(), blob.begin() + inputTensorSize, inputTensorValues_.begin());

    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, 
                                                           inputTensorValues_.data(), 
                                                           inputTensorValues_.size(), 
                                                           inputTensorShape_.data(), 
                                                           inputTensorShape_.size()));

    return inputTensors;
}

std::vector<Ort::Value> YOLOv8Detector::forward(std::vector<Ort::Value>& inputTensors) {
    return session_.Run(Ort::RunOptions{nullptr}, 
                        inputNamesCStr_.data(), 
                        inputTensors.data(), 
                        inputNamesCStr_.size(), 
                        outputNamesCStr_.data(), 
                        outputNamesCStr_.size());
}

std::vector<Detection> YOLOv8Detector::postprocess(std::vector<Ort::Value>& outputTensors, 
                                                float conf_threshold, 
                                                float iou_threshold) {
    float* data = outputTensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape(); // [1, 84, 8400]

    cv::Mat output0 = cv::Mat(static_cast<int>(outputShape[1]), static_cast<int>(outputShape[2]), CV_32F, data).t(); // (84, 8400) -> (8400, 84)

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float* pdata = (float*)output0.data;

    for (int i = 0; i < output0.rows; ++i) {
        cv::Mat scores(1, nc_, CV_32FC1, pdata + 4);
        double max_conf;
        cv::Point class_id;

        minMaxLoc(scores, nullptr, &max_conf, nullptr, &class_id);

        if (max_conf > conf_threshold) {
            class_ids.push_back(class_id.x);
            confidences.push_back(static_cast<float>(max_conf));
            
            float cx = pdata[0];
            float cy = pdata[1];
            float bw = pdata[2];
            float bh = pdata[3];

            float xmin = std::max(cx - bw * 0.5f, 0.0f);
            float ymin = std::max(cy - bh * 0.5f, 0.0f);

            cv::Rect_<float> bbox = cv::Rect_<float>(xmin, ymin, bw, bh);
            cv::Rect_<float> scaled_bbox = scale_boxes(bbox);

            boxes.push_back(scaled_bbox);
        }

        pdata += output0.cols; // ++84
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, nms_result);

    std::vector<Detection> detections;
    for (int i : nms_result) {
        boxes[i] &= cv::Rect(0, 0, orgImgSize_.width, orgImgSize_.height);

        Detection dect;
        dect.box = boxes[i];
        dect.classId = class_ids[i];
        dect.confidence = confidences[i];

        detections.push_back(dect);
    }

    return detections;
}

std::vector<Detection> YOLOv8Detector::infer(cv::Mat& frame, float conf_threshold, float iou_threshold) {
    std::vector<Ort::Value> inputTensors = preprocess(frame);
    std::vector<Ort::Value> outputTensors = forward(inputTensors);
    std::vector<Detection> detections = postprocess(outputTensors, conf_threshold, iou_threshold);

    return detections;
}

cv::Mat YOLOv8Detector::drawDetections(const cv::Mat& frame, const std::vector<Detection>& detections) {
    cv::Mat visImage = frame.clone(); // 원본 보존을 위해 복사본 생성
    for (const auto& detection : detections) {
        cv::rectangle(visImage, detection.box, cv::Scalar(0, 255, 0), 2);
    }
    return visImage;
}


std::unordered_map<int, std::string> YOLOv8Detector::parseNames(const std::string& input) {
    std::unordered_map<int, std::string> result;
    if (input.length() < 2) return result;

    size_t start = (input.front() == '{') ? 1 : 0;
    size_t end = (input.back() == '}') ? input.length() - 1 : input.length();
    std::string cleanedInput = input.substr(start, end - start);

    std::istringstream elementStream(cleanedInput);
    std::string element;
     // 0: 'Person', 1: 'Apple'
     // 쉼표(,)를 기준으로 라인을 가져옴
     // 0: 'Person'
     // 1: 'Apple'
    while (std::getline(elementStream, element, ',')) {
        size_t colon = element.find(':');
        if (colon != std::string::npos) { // npos: no position. 찾는 문자열이 없는 경우
            std::string keyStr = element.substr(0, colon); // 0
            std::string value = element.substr(colon + 1); // ' Person'

            // value에 작은따옴표나 공백이 포함되어 있다면 여기서 제거
            value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c){
                return c == '\'' || c == ' ';
            }), value.end()); // Person

            result[std::stoi(keyStr)] = value;
        }
    }

    return result;
}

cv::Mat YOLOv8Detector::letterbox(const cv::Mat& rgbFrame, cv::Scalar color, bool auto_, bool scalefill_, bool scaleUp, int stride) {
    cv::Mat outImage;
    // orgImgSize_ height: 640, width: 1280, r = 0.5
    scaleRatio_ = std::min( static_cast<float>(cvSize_.height) / static_cast<float>(orgImgSize_.height), 
                        static_cast<float>(cvSize_.width) / static_cast<float>(orgImgSize_.width) );
    if (!scaleUp) {
        scaleRatio_ = std::min(scaleRatio_, 1.0f);
    }

    // float ratio[2]{r_, r_};
    cv::Size resized{static_cast<int>( std::round(static_cast<float>(orgImgSize_.width) * scaleRatio_) ), // 640
                     static_cast<int>( std::round(static_cast<float>(orgImgSize_.height) * scaleRatio_) )}; // 320
    
    padX_ = cvSize_.width - resized.width; // 0
    padY_ = cvSize_.height - resized.height; // 320

    if (auto_){ // false
        padX_ = static_cast<float>(static_cast<int>(padX_) % stride); // 0
        padY_ = static_cast<float>(static_cast<int>(padY_) % stride); // 0
    }
    else if (scalefill_) { // false
        padX_ = 0.0f;
        padY_ = 0.0f;
        resized.width = cvSize_.width;
        resized.height = cvSize_.height;
        // ratio[0] = static_cast<float>(cvSize_.width) / static_cast<float>(orgImgSize_.width); // 0.5
        // ratio[1] = static_cast<float>(cvSize_.height) / static_cast<float>(orgImgSize_.height); // 1
    }

    padX_ /= 2.0f; // 0
    padY_ /= 2.0f; // 160

    if (cvSize_.width != resized.width || cvSize_.height != resized.height) { // 보정이 되었음
        cv::resize(rgbFrame, outImage, resized); // outImage는 width: 640, height: 320이 되었음.
    }
    else { // 원본 프레임이 640 640이라 보정이 필요없음. 그대로 clone 하면 됨.
        outImage = rgbFrame.clone();
    }

    int top = static_cast<int>(std::round(padY_ - 0.1f)); // 160
    int bottom = static_cast<int>(std::round(padY_ + 0.1f)); // 160
    int left = static_cast<int>(std::round(padX_ - 0.1f));
    int right = static_cast<int>(std::round(padX_ + 0.1f));

    if (color == cv::Scalar()) {
        color = cv::Scalar(DEFAULT_LETTERBOX_PAD_VALUE, DEFAULT_LETTERBOX_PAD_VALUE, DEFAULT_LETTERBOX_PAD_VALUE);
    }

    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return outImage;
}

std::vector<float> YOLOv8Detector::imageToBlob(cv::Mat& letterboxImage) {
    cv::Mat floatImage;
    letterboxImage.convertTo(floatImage, CV_32FC3, 1.0f / 255.0); // 정규화.

    std::vector<float> blob(floatImage.rows * floatImage.cols * floatImage.channels());
    cv::Size floatImageSize{floatImage.cols, floatImage.rows};

    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < chw.size(); ++i) {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob.data() + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
    return blob;
}

int64_t YOLOv8Detector::vector_product(const std::vector<int64_t>& shape) {
    int64_t result = 1;
    for (int64_t value : shape) {
        result *= value;
    }
    return result;
}


void YOLOv8Detector::clip_boxes(cv::Rect_<float>& box) {
    box.x = std::max(0.0f, std::min(box.x, static_cast<float>(orgImgSize_.width)));
    box.y = std::max(0.0f, std::min(box.y, static_cast<float>(orgImgSize_.height)));
    box.width = std::max(0.0f, std::min(box.width, static_cast<float>(orgImgSize_.width - box.x)));
    box.height = std::max(0.0f, std::min(box.height, static_cast<float>(orgImgSize_.height - box.y)));
}



cv::Rect_<float> YOLOv8Detector::scale_boxes(cv::Rect_<float>& box, bool padding) // true
{
    // r_ = 0.5, dw_ = 0, dh_ = 160
    cv::Rect_<float> scaledCoords(box);

    if (padding) {
        scaledCoords.x -= padX_; // xmin - 0
        scaledCoords.y -= padY_; // ymin - 160
    }

    scaledCoords.x /= scaleRatio_;
    scaledCoords.y /= scaleRatio_;
    scaledCoords.width /= scaleRatio_;
    scaledCoords.height /= scaleRatio_;

    clip_boxes(scaledCoords);

    return scaledCoords;
}





