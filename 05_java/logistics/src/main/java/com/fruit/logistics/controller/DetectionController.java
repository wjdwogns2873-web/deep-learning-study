package com.fruit.logistics.controller;

import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
@RequestMapping("/api/v1/detection")
public class DetectionController {

    @GetMapping("/health")
    public String healthCheck(){
        log.info("Health Check API 요청이 들어왔습니다.");
        return "Fruit Logistics Object Detection API Server is Running";
    }
}
