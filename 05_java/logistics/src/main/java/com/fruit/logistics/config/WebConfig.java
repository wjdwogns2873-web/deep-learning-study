package com.fruit.logistics.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {
    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        // /images/** 요청이 오면 프로젝트 루트의 uploads/ 폴더에서 파일을 찾아 보여줌
        registry.addResourceHandler("/images/**")
                .addResourceLocations("file:" + System.getProperty("user.dir") + "/uploads/");
    }
}