package com.fruit.logistics.domain;

import jakarta.persistence.Entity;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

//import javax.persistence.*;
import jakarta.persistence.*; // Correct
import java.time.LocalDateTime;

@Entity
@Getter
@NoArgsConstructor
@Table(name = "detection_history")
public class DetectionHistory {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String originalFileName; // 원본 파일명
    private String savedFileName; // 바운딩 박스가 그려져 저장된 파일명
    private String detectedFruit; // 탐지된 과일
    private Double confidence; // 확신도
    private String bboxCoordinates; // 좌표
    private Double conf_threshold; // 확신도 임계값
    private Double iou_threshold; // IoU 임계값

    private LocalDateTime createdDate; // 탐지 시각

    @Builder
    public DetectionHistory(String originalFileName,
                            String savedFileName,
                            String detectedFruit,
                            Double confidence,
                            String bboxCoordinates,
                            Double conf_threshold,
                            Double iou_threshold) {
        this.originalFileName = originalFileName;
        this.savedFileName = savedFileName;
        this.detectedFruit = detectedFruit;
        this.confidence = confidence;
        this.bboxCoordinates = bboxCoordinates;
        this.conf_threshold = conf_threshold;
        this.iou_threshold = iou_threshold;
        this.createdDate = LocalDateTime.now();
    }
}
