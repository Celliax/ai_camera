## AI 쓰레기 인식

세월e가 수거한 쓰레기를 카메라로 촬영해 종류를 자동으로 판별하고, 그 결과에 따라 분류 팔을 움직이는 부분이다.
휴지와 종이, 페트병과 플라스틱처럼 사람도 헷갈리는 쓰레기를 구분하는 것을 목표로 한다.

### 개요

| 항목 | 내용 |
|---|---|
| 모델 | YOLOv8s (Ultralytics) |
| 분류 클래스 | 병 · 플라스틱 · 종이 · 일반쓰레기(휴지) |
| 학습 데이터 | AI Hub 생활폐기물 이미지 + 휴지 데이터 (학습 약 13만 장, 검증 약 1.6만 장) |
| 학습 환경 | RTX 4060 노트북 GPU, 입력 320, 30 epoch, COCO 사전학습 가중치에서 전이학습 |
| 추론 환경 | Jetson Orin Nano, TensorRT FP16 엔진, Docker |
| 카메라 | Gevisun 1080P USB 웹캠 |

### 동작 흐름

```
USB 웹캠 → 320 크기로 전처리 → TensorRT 추론 (Jetson)
        → 신뢰도 0.4 이상 결과만 채택
        → 웹 화면 실시간 전송 (Flask-SocketIO)
        → 아두이노에 시리얼 명령 → 분류 팔 동작
```

같은 쓰레기로 팔이 여러 번 움직이지 않도록, 쓰레기가 여러 프레임 연속으로 탐지됐을 때만 팔을 움직이고 동작 중과 동작 직후에는 새 탐지를 무시한다.

### 성능

학습에 쓰지 않은 검증 이미지 16,305장 기준이다.

| 지표 | 값 |
|---|---|
| mAP50 | 0.977 |
| mAP50-95 | 0.924 |
| 정밀도 (Precision) | 0.965 |
| 재현율 (Recall) | 0.949 |

YOLOv8n에서 YOLOv8s로 모델을 키워 일반쓰레기 정밀도가 0.849에서 0.949로, 플라스틱 재현율이 0.898에서 0.952로 올랐다.
검증셋은 학습 데이터와 같은 출처이므로 실제 카메라 환경의 성능은 별도로 검증이 필요하다.

### 실행 방법 (Jetson)

필요한 것:
- 홈 폴더에 `best.engine` (TensorRT 엔진)
- `walle` Docker 이미지
- `pip3 install "python-socketio[client]" websocket-client requests pyserial`

```bash
python3 ai_camera.py                 # 카메라 + AI 추론 + 분류 팔 전체 실행
python3 ai_camera.py --no-arm        # 분류 팔 없이 카메라와 AI만 실행
python3 ai_camera.py --arm-test      # 분류 팔만 한 번 시험
python3 ai_camera.py --list-ports    # 아두이노 포트 확인
```

실행하면 같은 네트워크의 브라우저에서 `http://<Jetson IP>:5000` 으로 원본 화면과 탐지 화면을 볼 수 있다.

주요 옵션:

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--conf` | 탐지 신뢰도 기준 | 0.4 |
| `--camera` | 카메라 종류 (auto / usb / csi) | auto |
| `--arm-confirm` | 팔이 움직이기 위한 연속 탐지 프레임 수 | 3 |
| `--arm-delay` | 탐지 후 팔 동작까지 기다리는 시간(초) | 0 |
| `--arm-cooldown` | 팔 동작 후 쉬는 시간(초) | 2 |
| `--arm-classes` | 특정 클래스일 때만 팔 동작 (쉼표 구분) | 전체 |

### 모델 학습 및 배포

1. 노트북에서 `train_v3.py`로 학습한다. 학습이 끝나면 기존 모델과 같은 검증셋으로 자동 비교한다.
2. 학습된 `best.pt`를 Jetson으로 복사한다.
3. Jetson에서 TensorRT 엔진으로 변환한다. 엔진은 만든 기기에서만 동작하므로 반드시 Jetson에서 변환한다.

```bash
yolo export model=best.pt format=engine half=True imgsz=320
```
