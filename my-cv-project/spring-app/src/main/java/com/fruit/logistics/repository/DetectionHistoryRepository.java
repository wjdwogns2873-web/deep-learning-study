package com.fruit.logistics.repository;

import com.fruit.logistics.domain.DetectionHistory;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface DetectionHistoryRepository extends JpaRepository<DetectionHistory, Long> {
    // 최신순으로 탐지 이력 조회
    List<DetectionHistory> findAllByOrderByIdDesc();
}