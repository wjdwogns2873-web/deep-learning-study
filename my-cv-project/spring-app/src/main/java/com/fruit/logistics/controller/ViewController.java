package com.fruit.logistics.controller;

import com.fruit.logistics.service.DetectionService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
@RequiredArgsConstructor
public class ViewController {
    private final DetectionService detectionService;

    // 메인 웹 대시보드 페이지 (http://localhost:8080/)
    @GetMapping("/")
    public String index(Model model) {
        // DB에 저장된 탐지 이력 목록을 Model에 담아 HTML로 전달
        model.addAttribute("histories", detectionService.getAllHistories());
        return "index"; // templates/index.html 파일 호출
    }
}
