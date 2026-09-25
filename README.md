# 컴퓨터 비전 엔지니어 정재훈의 기술 저장소

2년 6개월간의 자바 백엔드 개발 경험과 유니티(C#) 경험을 결합하여, 합성 데이터(Synthetic Data) 생성부터 **C++ Drogon / Python 고성능 AI 추론 파이프라인 최적화**까지 구축하는 컴퓨터 비전 엔지니어입니다.

🌐 **포트폴리오 Web**: http://jaehun-cv.duckdns.org/

---

### 핵심 역량
* **3D 합성 데이터(Synthetic Data) 생성**: Unity(C#) 기반 3D 합성 데이터 및 바운딩 박스 좌표 투영 연산 구축
* **AI 모델 배포 및 추론 최적화**: PyTorch, ONNX Runtime, TensorRT 추론 엔진 비교 및 최적화 [(👉 벤치마크 보고서 보기)](https://github.com/wjdwogns2873-web/deep-learning-study/blob/main/09_%EB%B2%A4%EC%B9%98%EB%A7%88%ED%81%AC_%EB%B3%B4%EA%B3%A0%EC%84%9C/TensorRT_%EC%A0%9C%EB%A1%9C%EC%B9%B4%ED%94%BC_%EB%AA%A8%EB%8D%B8_%EC%B6%94%EB%A1%A0%EC%86%8D%EB%8F%84.md)
* **C++ 고성능 추론 엔진 구축**: C++, TensorRT로 추론 성능을 극한으로 끌어올림
* **고성능 비동기 파이프라인 설계**: FastAPI / Drogon WebSocket 기반 바이너리 프로토콜 및 Queue (producer-consumer) 비동기 파이프라인 설계

### 🎮 Elevator (3D Action)

<p align="center">
  <img src="https://github.com/user-attachments/assets/868a39af-db60-4e90-b4c4-f8c57fe628e1" width="100%" alt="Elevator">
</p>

<details>
<summary><b>💡 게임 기획 배경, 규칙 및 특징 (클릭하여 펼치기)</b></summary>

<br>

* **기획 배경**
  * 매일 아침 출근길 엘리베이터를 타며 *"승객들의 목적지 층수 버튼을 플레이어가 직접 외워 누른다면 어떨까?"*라는 아이디어에서 출발했습니다.

* **게임 규칙**
  * **승객 탑승**: 승객이 타면 머리 위 말풍선에 목적지 층수가 잠시 표시됩니다.
  * **기억 & 조작**: 플레이어는 말풍선이 사라지기 전에 층수를 기억하여 우측 엘리베이터 버튼을 눌러야 합니다.
  * **게임 오버**: 좌측 상단 하트(목숨 2개)가 모두 소진되면 Game Over됩니다.

* **개발 포인트 및 성과**
  * **3D 공간 좌표계 & 캐릭터 승하차 이동 로직** 직접 구현
  * 퀄리티를 높이기 위해 유료 에셋 구매 및 무료 엘리베이터 효과음 사용
  * 출시된 3개 유니티 프로젝트 중 **가장 준수한 퀄리티와 3D 구현 경험**을 담은 핵심 프로젝트

</details>

---

<table width="100%">
  <tr>
    <td align="center" width="50%" valign="top">
      <h3>🔴 Red Ball OUT!</h3>
      <img src="https://github.com/user-attachments/assets/d1f1d844-082e-418e-83ce-a024473b28e3" width="75%" alt="Red Ball Out"><br><br>
      <details>
        <summary><b>💡 기획 배경 및 규칙</b></summary>
        <br>
        <div align="left">
          • <b>기획 배경</b>: 직관적인 룰을 구현하기 위한 아이디어<br>
          • <b>게임 규칙</b>: 하단 슬라이드 바 조작으로 상자 길이를 조절하여 빨간 공만 바깥으로 배출 (파란 공 배출 시 Game Over)<br>
          • <b>개발 성과</b>: 유니티 2D 물리 엔진 숙달을 통해 <b>기획부터 출시까지 1~2주 만에 신속 완수</b>
        </div>
      </details>
    </td>
    <td align="center" width="50%" valign="top">
      <h3>🟦 Make a Square!</h3>
      <img src="https://github.com/user-attachments/assets/6d33fdaa-7551-4233-a664-2583f8fbc0e8" width="75%" alt="Make a Square"><br><br>
      <details>
        <summary><b>💡 기획 배경 및 규칙</b></summary>
        <br>
        <div align="left">
          • <b>기획 배경</b>: 타이밍 포착 기반의 정사각형 완성 아이디어<br>
          • <b>게임 규칙</b>: 변하는 선을 포착하여 화면 터치로 정사각형 생성 (크기가 클수록 고득점, 불일치 시 Game Over)<br>
          • <b>개발 성과</b>: **첫 구글 플레이스토어 출시작**으로, 이후 프로젝트 출시를 위한 기초 기반 마련
        </div>
      </details>
    </td>
  </tr>
</table>

[게임 아이디어 노트](https://github.com/wjdwogns2873-web/deep-learning-study/tree/main/game_idea_note)

