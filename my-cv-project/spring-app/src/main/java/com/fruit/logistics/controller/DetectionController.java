package com.fruit.logistics.controller;

import com.fruit.logistics.domain.DetectionHistory;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;

import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.*;
//import org.springframework.web.bind.annotation.GetMapping;
//import org.springframework.web.bind.annotation.RequestMapping;
//import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import com.fruit.logistics.service.DetectionService;
import lombok.RequiredArgsConstructor;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

@Slf4j
@RestController
@RequestMapping("/api/v1/detection")
@RequiredArgsConstructor // 생성자 주입을 통한 Service 자동 연결 (@Autowired 대용)
public class DetectionController {
    private final DetectionService detectionService; // Service 주입

    // 과일 이미지 파일을 받아 객체 탐지(ai 모델 연동)를 요청하는 엔드포인트
    @PostMapping(value = "/detect")
    public ResponseEntity<byte[]> detectObject(@RequestParam("file") MultipartFile file,
                                                            @RequestParam("conf_value") double conf_value,
                                                            @RequestParam("iou_value") double iou_value){
        log.info("[이미지 수신 완료] 파일명: {}", file != null ? file.getOriginalFilename() : "없음");

        byte[] imageBytes = detectionService.requestObjectDetection(file, conf_value, iou_value);

//        return ResponseEntity.ok(detectionResult);
        return ResponseEntity.ok()
                .contentType(MediaType.IMAGE_JPEG)
                .body(imageBytes);
    }

    // 탐지 이력 조회
    @GetMapping("/histories")
    public ResponseEntity<List<DetectionHistory>> getHistory() {
        List<DetectionHistory> histories = detectionService.getAllHistories();
        return ResponseEntity.ok(histories);
    }


}
