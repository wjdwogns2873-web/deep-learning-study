package com.fruit.logistics;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@Slf4j
@SpringBootTest // 이 어노테이션이 스프링 부트 로깅 엔진을 켜줍니다.
class FruitApplicationTests {

	@Test
	void logginTest() {
		System.out.println("=== 1. println 출력 확인 ===");
		log.info("=== 2. log.info 출력 확인 완료!!! ===");
	}
}