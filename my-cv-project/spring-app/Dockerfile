# 1단계: Gradle 환경에서 자바 소스코드 빌드
FROM gradle:8.5-jdk17 AS builder
WORKDIR /app

# 빌드에 필요한 파일들 복사
COPY build.gradle settings.gradle gradlew ./
COPY gradle ./gradle
COPY src ./src

# 도커 내부에서 직접 jar 파일 빌드 (테스트 제외)
RUN ./gradlew build -x test


# 2단계: 생성된 jar 파일만 가져와서 실행
FROM eclipse-temurin:17-jdk
WORKDIR /app

# 1단계(builder)에서 빌드된 jar 파일 복사
COPY --from=builder /app/build/libs/*.jar app.jar

EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]