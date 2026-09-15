package com.fruit.logistics.service;

import com.fruit.logistics.domain.DetectionHistory;
import com.fruit.logistics.repository.DetectionHistoryRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Base64;
import java.util.UUID;
/*
이미지 탐지: http://localhost:8000 + /predict/image
동영상 스트리밍: http://localhost:8000 + /stream/video
* */
@Slf4j
@Service
@RequiredArgsConstructor
public class DetectionService {
    @Value("${ai.server.url}") // http://localhost:8000
    private String aiServerUrl;

    public byte[] requestObjectDetection(MultipartFile file,
                                         double conf_value,
                                         double iou_value,
                                         String model_type) {

        if (file == null || file.isEmpty()) {
            log.warn("업로드된 이미지 파일이 존재하지 않거나 빈 파일입니다.");
            throw new IllegalArgumentException("이미지 파일을 선택해 주세요.");
        }

        RestTemplate restTemplate = new RestTemplate();
        String fullUrl = aiServerUrl + "/predict/image";

        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            // MultipartFile -> ByteArrayResource 변환
            ByteArrayResource fileAsResource = new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            };

            // Request Body 구성
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", fileAsResource);
            body.add("conf_value", conf_value);
            body.add("iou_value", iou_value);
            body.add("model_type", model_type);

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            log.info("AI 서버({})로 요청 전송 [Engine: {}]", fullUrl, model_type);

            // FastAPI 서버로 POST 요청
            ResponseEntity<byte[]> response = restTemplate.postForEntity(fullUrl, requestEntity, byte[].class);

            return response.getBody();

        } catch (IOException e) {
            log.error("파일 변환 중 오류 발생: {}", e.getMessage());
            throw new RuntimeException("이미지 파일 처리 실패");
        } catch (Exception e) {
            log.error("파이썬 AI 서버 통신 에러: {}", e.getMessage());
            throw new RuntimeException("AI 서버 연동 실패 - 파이썬 FastAPI 서버 실행 여부를 확인하세요.");
        }
    }

}
