package com.fruit.logistics;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@Slf4j
@SpringBootApplication
public class LogisticsApplication {

	public static void main(String[] args) {
		SpringApplication.run(LogisticsApplication.class, args);
		log.info("Hello Worlddddddd");
		System.out.println("fff");
	}

}
