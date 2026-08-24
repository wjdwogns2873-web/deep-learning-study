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

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
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
    private final DetectionHistoryRepository historyRepository; // Repository 주입
    private final String UPLOAD_DIR = System.getProperty("user.dir") + "/uploads/";

    // 수신받은 이미지를 파이썬 AI 서버로 전송하고 결과를 받아오는 메서드
    public Map<String, Object> requestObjectDetection(MultipartFile file,
                                                      double conf_value,
                                                      double iou_value) {
        RestTemplate restTemplate = new RestTemplate();

        String suffix_url = "/predict/image";
        String full_url = aiServerUrl + suffix_url;

        try {
            // HTTP Header 설정 (multipart/form-data)
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            // MultipartFile을 파이썬으로 보낼 수 있는 Resource 형태로 변환
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

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

            log.info("파이썬 AI 서버({})로 이미지 전송 요청 중..", full_url);

            // 파이썬 서버로 POST 요청 전송 (현재는 파이썬 서버가 안 켜져 있으므로 예외처리 준비)
            ResponseEntity<Map> response = restTemplate.postForEntity(full_url, requestEntity, Map.class);
            Map<String, Object> responseBody = response.getBody();

            // DB에 탐지 결과 저장 로직 추가
            if (responseBody != null && "SUCCESS".equals(responseBody.get("status"))) {
                // 파이썬에서 받은 Base64 결과 이미지 바이트로 디코딩
                String base64Image = (String) responseBody.get("result_image_base64");
                String savedFileName = saveBase64Image(base64Image);// 파일 저장 후 저장된 파일명 리턴

                List<Map<String, Object>> predictions = (List<Map<String, Object>>) responseBody.get("predictions");

                for (Map<String, Object> pred : predictions) {
                    String label = (String) pred.get("label");
                    Double confidence = Double.valueOf(pred.get("confidence").toString());
                    String boxStr = pred.get("box").toString();

                    DetectionHistory history = DetectionHistory.builder()
                            .originalFileName(file.getOriginalFilename())
                            .savedFileName(savedFileName)
                            .detectedFruit(label)
                            .confidence(confidence)
                            .bboxCoordinates(boxStr)
                            .conf_threshold(conf_value)
                            .iou_threshold(iou_value)
                            .build();

                    historyRepository.save(history); // DB Insert
                    log.info("DB 저장 완료: 과일: {}, 신뢰도: {}", label, confidence);
                }
            }

            log.info("파이썬 AI 서버 응답 수신 완료");
            return responseBody;

        } catch (IOException e) {
            log.error("파일 변환 중 오류 발생: {}", e.getMessage());
            throw new RuntimeException("이미지 파일 처리 실패", e);
        } catch (Exception e) {
            log.error("파이썬 AI 서버 통신 에러: {}", e.getMessage());
            throw new RuntimeException("AI 서버 연동 실패 - 파이썬 서버가 켜져 있는지 확인하세요");
        }
    }

    // Base64 문자열을 실제 .jpg 파일로 uploads 폴더에 저장하는 유틸 메서드
    private String saveBase64Image(String base64Image) {
        try {
            File dir = new File(UPLOAD_DIR);
            if (!dir.exists()) {
                dir.mkdirs();
            }

            byte[] imageBytes = Base64.getDecoder().decode(base64Image);
            String savedFileName = "detected_" + UUID.randomUUID().toString().substring(0, 8) + ".jpg";
            File targetFile = new File(UPLOAD_DIR + savedFileName);

            try (FileOutputStream fos = new FileOutputStream(targetFile)) {
                fos.write(imageBytes);
            }

            log.info("결과 이미지 로컬 저장 완료: {}", targetFile.getAbsolutePath());
            return savedFileName;
        } catch (Exception e) {
            log.error("이미지 저장 실패: {}", e.getMessage());
            return null;
        }
    }

    // 전체 탐지 이력 목록 조회 메서드
    public List<DetectionHistory> getAllHistories() {
        return historyRepository.findAllByOrderByIdDesc();
    }

}
