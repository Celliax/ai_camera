#!/usr/bin/env python3
"""
AI 카메라 런처 (Jetson 호스트에서 실행)

이 파일 하나만 실행하면:
  1) walle 컨테이너를 띄워서 YOLO(TensorRT) 추론 서버를 시작하고
  2) 카메라(CSI 또는 USB, 자동 감지)를 열어서 프레임을 서버로 보내고
  3) 같은 네트워크의 다른 노트북 브라우저에서 원본/탐지 화면을 볼 수 있고
  4) 쓰레기가 탐지되면 아두이노 분류 팔을 자동으로 움직입니다.

사용법 (호스트 터미널, 즉 juni@ubuntu 프롬프트에서):
    python3 ai_camera.py

종료: Ctrl+C  (컨테이너도 같이 정리됩니다)

카메라 선택 (기본은 자동 감지: CSI 카메라가 보이면 CSI, 없으면 USB 카메라):
    python3 ai_camera.py --camera usb              USB 카메라로 강제
    python3 ai_camera.py --camera csi              CSI 카메라로 강제
    python3 ai_camera.py --camera usb --device 2   USB 장치 번호 지정 (/dev/video2)
    python3 ai_camera.py --flip 2                  화면 뒤집기 (0 없음, 2 180도, 4 좌우, 6 상하)
탐지가 잘 안 잡힐 때 시험해볼 옵션:
    python3 ai_camera.py --conf 0.3        신뢰도 기준을 낮춤 (기본 0.4)
    python3 ai_camera.py --send-fps 30     서버로 보내는 최대 장수 (기본 20)
    ※ 색 번짐/렌즈 왜곡 보정 파일(shading_gain.npy, lens_maps.npz)은 CSI(imx219) 카메라용이라
      USB 모드에서는 쓰지 않습니다. (USB 렌즈 보정을 하면 lens_maps_usb.npz 로 따로 저장됨)

분류 팔 (아두이노):
    기본으로 아두이노를 자동으로 찾아서 연결합니다. 못 찾으면 팔 없이 카메라만 실행됩니다.
    쓰레기가 --arm-confirm 프레임 연속으로 탐지되면 팔을 한 번 움직이고(약 6초),
    동작 중과 동작 직후(--arm-cooldown 초)에는 새로 움직이지 않습니다.
    python3 ai_camera.py --list-ports              시리얼 포트 목록만 보고 종료
    python3 ai_camera.py --arm-test                팔을 한 번 움직여 보고 종료 (카메라/서버 없이)
    python3 ai_camera.py --arm-port /dev/ttyACM0   아두이노 포트 직접 지정
    python3 ai_camera.py --no-arm                  팔 없이 카메라만 실행
    python3 ai_camera.py --arm-confirm 5           5프레임 연속 탐지돼야 동작 (기본 3)
    python3 ai_camera.py --arm-cooldown 3          동작이 끝난 뒤 3초 쉼 (기본 2)
    python3 ai_camera.py --arm-delay 1.5           탐지 후 1.5초 기다렸다가 동작 (기본 0, 바로 동작)
    python3 ai_camera.py --arm-classes 일반쓰레기   이 클래스가 탐지될 때만 동작 (쉼표로 여러 개, 기본 전체)
    ※ 아두이노 IDE 의 시리얼 모니터가 열려 있으면 포트를 쓸 수 없으니 닫으세요.
    ※ "Permission denied" 가 나오면: sudo usermod -aG dialout $USER  (그 뒤 재로그인 또는 재부팅)

렌즈 쉐이딩(중앙 초록/가장자리 분홍 색 번짐) 보정:
    1) 렌즈 위에 흰 종이를 2~3겹 꽉 덮고 밝은 조명 쪽을 향하게 둔다
    2) python3 ai_camera.py --calibrate     (보정 지도를 ~/shading_gain.npy 로 저장)
    3) 종이를 치우고 평소처럼 python3 ai_camera.py 실행 (보정이 자동 적용됨)
    보정을 끄고 싶으면: python3 ai_camera.py --no-shading  (또는 shading_gain.npy 삭제)

렌즈 왜곡(광각 렌즈 때문에 화면이 볼록하게 휘는 것) 보정:
    1) 체스보드 이미지(chessboard_9x6.png)를 노트북 화면에 크게 띄운다
    2) python3 ai_camera.py --lens-calibrate   (평소처럼 실행되면서 체스보드를 자동 수집)
    3) 브라우저로 화면을 보면서 체스보드를 카메라 앞에서 위치/각도/거리를 바꿔가며 보여준다
       (터미널에 수집 횟수가 올라가고, 다 모이면 자동 계산해서 ~/lens_maps.npz 로 저장 + 바로 적용)
    4) 다음부터는 평소처럼 실행하면 자동 적용. 끄고 싶으면 --no-lens (또는 lens_maps.npz 삭제)

체스보드 없이 렌즈 왜곡 보정 (눈으로 맞추기):
    1) python3 ai_camera.py --lens-manual
    2) 브라우저 화면을 보면서, 다른 호스트 터미널에서:  echo 150 > ~/lens_fov.txt
       직선(벽 모서리, 천장 선)이 곧게 보일 때까지 숫자를 바꾼다 (바꾸면 1초 안에 반영, 자동 저장)
    3) 다음부터는 평소처럼 실행하면 저장된 값이 자동 적용됨
    ※ 렌즈 보정 모드(--lens-calibrate, --lens-manual)에서는 분류 팔을 쓰지 않습니다.

필요 조건:
  - /home/juni/best.engine 이 있어야 함
  - 호스트에 패키지: pip3 install "python-socketio[client]" websocket-client requests pyserial
    (numpy, cv2 는 이미 있을 가능성이 높음. pyserial 은 분류 팔을 쓸 때만 필요)
"""

import argparse
import base64
import glob
import math
import os
import queue
import re
import select
import socket
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import socketio

try:
    import serial
    from serial.tools import list_ports
except ImportError:          # 분류 팔을 안 쓰면 pyserial 이 없어도 실행되게
    serial = None
    list_ports = None

# ── 설정 ──────────────────────────────────────────────
HOME = os.path.expanduser("~")
ENGINE_FILE = os.path.join(HOME, "best.engine")
SERVER_FILE = os.path.join(HOME, "_yolo_server.py")   # 아래 SERVER_CODE 가 여기에 저장됨

DOCKER_IMAGE = "walle"
CONTAINER_NAME = "yolo_server"
SERVER_PORT = 5000

SENSOR_ID = 0
CAPTURE_W, CAPTURE_H = 1640, 1232   # imx219 4:3 모드 (30fps, 화각 넓음)
FPS = 30
OUT_W, OUT_H = 640, 480             # 서버로 보낼 해상도
FLIP_METHOD = 0                     # 화면이 뒤집혀 보이면 2(180도 회전)로 변경
JPEG_QUALITY = 90                   # 모델이 보는 영상의 화질 (같은 Jetson 안(localhost)으로 보내므로 높여도 부담 없음)
SEND_INTERVAL = 0.05                # 최대 약 20 FPS 로 전송 (--send-fps 로 변경)
DEFAULT_CONF = 0.4                  # 탐지 신뢰도 기준 (--conf 로 변경)

FRAME_BYTES = OUT_W * OUT_H * 3

# ── 분류 팔(아두이노) 설정 ────────────────────────────
ARM_BAUD = 9600                     # 스케치의 Serial.begin(9600) 과 같아야 함
ARM_CONFIRM = 3                     # 이만큼 연속 프레임에서 탐지돼야 팔을 움직임 (한 프레임 오탐 방지)
ARM_COOLDOWN = 2.0                  # 팔 동작이 끝난 뒤 쉬는 시간(초)
ARM_DELAY = 0.0                     # 탐지 확정 후 팔을 움직이기 전까지 기다리는 시간(초)

# 자동 감지에 쓰는 USB 제조사 번호: 정품 아두이노, CH340 호환 보드.
# CP210x(0x10C4), FTDI(0x0403) 는 라이다 같은 다른 장치에도 많이 쓰여서 일부러 뺐음 (--arm-port 로 직접 지정)
AUTO_DETECT_VIDS = {0x2341, 0x2A03, 0x1A86}

# ── 렌즈 쉐이딩 보정 설정 ─────────────────────────────
GAIN_FILE = os.path.join(HOME, "shading_gain.npy")
CALIB_SKIP = 20          # 보정 촬영 시작 직후 노출/화이트밸런스가 안정될 때까지 버릴 프레임 수
CALIB_FRAMES = 40        # 평균낼 프레임 수
LUMA_STRENGTH = 0.5      # 가장자리 밝기 보정 세기 (0 = 색만 보정, 1 = 밝기도 완전 보정, 클수록 노이즈 증가)

# ── 렌즈 왜곡 보정 설정 ───────────────────────────────
LENS_FILE = os.path.join(HOME, "lens_maps.npz")
CHECKERBOARD = (9, 6)    # 체스보드 안쪽 모서리 개수 (가로, 세로) - chessboard_9x6.png 기준
LENS_VIEWS = 20          # 수집할 체스보드 장면 수
LENS_INTERVAL = 0.7      # 체스보드 탐색 간격(초)
LENS_MIN_MOVE = 25       # 이전 수집분과 이만큼(px) 이상 달라야 새 장면으로 인정
LENS_BALANCE = 0.0       # 0 = 검은 테두리 없이 꽉 차게(가장자리 시야는 일부 잘림), 1 = 시야 최대 유지(검은 테두리 생김)
LENS_FOV_FILE = os.path.join(HOME, "lens_fov.txt")  # --lens-manual 모드: 이 파일의 숫자(렌즈 대각 시야각, 도)를 바꾸면 바로 반영
LENS_FOV_DEFAULT = 140.0

# ── 컨테이너 안에서 돌아갈 추론 서버 코드 ─────────────────
SERVER_CODE = r"""
from flask import Flask
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import cv2
import base64
import numpy as np
import threading
import os
import time

MODEL_PATH = "/workspace/best.engine"
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.4"))
IMGSZ = 320

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10_000_000)
model = YOLO(MODEL_PATH)
print("YOLO 모델 로드 완료!", flush=True)
print(f"신뢰도 기준 conf={CONF_THRESHOLD}", flush=True)

latest_frame = None
frame_lock = threading.Lock()


@socketio.on('send_frame')
def handle_frame(data):
    global latest_frame
    try:
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        _, raw = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        emit('raw_frame', base64.b64encode(raw).decode(), broadcast=True)
        with frame_lock:
            latest_frame = frame
    except Exception as e:
        print(f"{e}", flush=True)


def inference_loop():
    global latest_frame
    count = 0
    t0 = time.time()
    while True:
        frame = None
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame
                latest_frame = None
        if frame is None:
            time.sleep(0.005)
            continue
        try:
            results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)
            r = results[0]
            annotated = r.plot()
            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 50])
            socketio.emit('detection_frame', base64.b64encode(buf).decode())

            # 탐지 결과(클래스 이름, 신뢰도)를 호스트로 보냄 -> 호스트가 분류 팔을 움직임
            dets = []
            if r.boxes is not None and len(r.boxes):
                for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    dets.append({"cls": r.names[int(c)], "conf": round(float(s), 3)})
            socketio.emit('detections', dets)

            count += 1
            if count % 50 == 0:
                now = time.time()
                print(f"추론 속도: {50 / (now - t0):.1f} FPS", flush=True)
                t0 = now
        except Exception as e:
            print(f"추론 오류: {e}", flush=True)


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jetson 쓰레기 탐지</title>
        <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
        <style>
            body { font-family: Arial; text-align: center; padding: 20px; background: #f0f0f0; }
            .container { background: white; padding: 30px; border-radius: 10px; max-width: 850px; margin: 0 auto; }
            .views { display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; margin-top: 20px; }
            .view-box { flex: 1; min-width: 320px; }
            img { max-width: 100%; border: 2px solid #333; border-radius: 4px; }
            #status { font-size: 18px; font-weight: bold; }
            .connected { color: green; }
            .disconnected { color: red; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Jetson 카메라 + YOLO</h1>
            <p id="status" class="disconnected">연결 중...</p>
            <div class="views">
                <div class="view-box">
                    <h3>실시간 카메라 (원본)</h3>
                    <img id="rawFrame" src="" style="width: 320px;">
                </div>
                <div class="view-box">
                    <h3>탐지 결과 (YOLO)</h3>
                    <img id="detectionFrame" src="" style="width: 320px;">
                </div>
            </div>
        </div>
        <script>
            const socket = io();
            socket.on('connect', () => {
                document.getElementById('status').innerHTML = 'Jetson 준비됨';
                document.getElementById('status').className = 'connected';
            });
            socket.on('raw_frame', (d) => {
                document.getElementById('rawFrame').src = 'data:image/jpeg;base64,' + d;
            });
            socket.on('detection_frame', (d) => {
                document.getElementById('detectionFrame').src = 'data:image/jpeg;base64,' + d;
            });
            socket.on('disconnect', () => {
                document.getElementById('status').innerHTML = '연결 끊김';
                document.getElementById('status').className = 'disconnected';
            });
        </script>
    </body>
    </html>
    '''


if __name__ == '__main__':
    threading.Thread(target=inference_loop, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
"""

def build_gst_cmd(flip):
    return [
        "gst-launch-1.0", "-q",
        "nvarguscamerasrc", f"sensor-id={SENSOR_ID}", "!",
        f"video/x-raw(memory:NVMM),width={CAPTURE_W},height={CAPTURE_H},framerate={FPS}/1,format=NV12", "!",
        "nvvidconv", f"flip-method={flip}", "!",
        f"video/x-raw,width={OUT_W},height={OUT_H},format=BGRx", "!",
        "videoconvert", "!",
        "video/x-raw,format=BGR", "!",
        "fdsink", "fd=1",
    ]


GST_CMD = build_gst_cmd(FLIP_METHOD)  # --calibrate(색 번짐 보정)가 사용. main() 에서 --flip 값으로 갱신될 수 있음

sio = socketio.Client()
frame_queue = queue.Queue(maxsize=1)  # 최신 프레임만 유지


@sio.event
def connect():
    print("추론 서버에 연결됨!")


@sio.event
def disconnect():
    print("추론 서버 연결 끊김")


# ── 분류 팔 (아두이노, 시리얼) ─────────────────────────
# 아두이노 스케치의 약속(프로토콜):
#   - 시리얼 연결 직후 "READY" 를 한 번 보냄 (포트를 열면 보드가 재시작됨)
#   - 'T' 를 받으면 서보 A, B 를 차례로 왕복시키고 (약 6초) "DONE" 을 보냄
#   - 동작 중에 받은 명령은 수신 버퍼에 쌓였다가 연달아 실행됨 (그래서 동작 중 중복 전송을 막아야 함)

class ArmError(Exception):
    """팔 제어 중 발생한 오류 (연결 실패, 시리얼 끊김 등)"""


def _require_pyserial():
    if serial is None:
        raise ArmError("pyserial 이 없습니다. 설치: pip3 install pyserial  (팔 없이 실행하려면 --no-arm)")


def list_serial_ports():
    _require_pyserial()
    return list(list_ports.comports())


def find_arduino_port():
    """아두이노로 보이는 포트를 찾는다. 하나도 없으면 None, 여러 개면 ArmError."""
    candidates = [p for p in list_serial_ports() if p.vid in AUTO_DETECT_VIDS]
    if not candidates:
        return None
    if len(candidates) > 1:
        names = ", ".join(p.device for p in candidates)
        raise ArmError(f"아두이노로 보이는 포트가 여러 개입니다 ({names}). --arm-port 로 직접 지정하세요.")
    return candidates[0].device


def print_serial_ports():
    try:
        ports = list_serial_ports()
    except ArmError as e:
        print(e)
        return
    if not ports:
        print("연결된 시리얼 포트가 없습니다.")
        return
    for p in ports:
        vid = f"{p.vid:04X}" if p.vid is not None else "----"
        pid = f"{p.pid:04X}" if p.pid is not None else "----"
        mark = "  <- 아두이노로 보임" if p.vid in AUTO_DETECT_VIDS else ""
        print(f"{p.device}  [{vid}:{pid}]  {p.description}{mark}")


class ArmController:
    def __init__(self, port=None, baud=ARM_BAUD, ready_timeout=8.0, done_timeout=20.0, log=print):
        """
        port           시리얼 포트 (None 이면 자동 감지)
        baud           통신 속도 (스케치의 Serial.begin(9600) 과 같아야 함)
        ready_timeout  연결 후 READY 를 기다리는 최대 시간(초)
        done_timeout   명령 후 DONE 을 기다리는 최대 시간(초). 한 번 동작이 약 6초
        log            메시지를 출력하는 함수
        """
        self.port = port
        self.baud = baud
        self.ready_timeout = ready_timeout
        self.done_timeout = done_timeout
        self._log = log
        self._ser = None
        self._port_in_use = None
        self._busy_lock = threading.Lock()   # 동작 중에는 잠겨 있음

    def connect(self):
        _require_pyserial()
        port = self.port or find_arduino_port()
        if not port:
            raise ArmError("아두이노를 찾지 못했습니다. 연결을 확인하거나 --arm-port 로 직접 지정하세요. "
                           "(python3 ai_camera.py --list-ports 로 포트 목록 확인)")
        self._port_in_use = port
        try:
            self._ser = serial.Serial(port, self.baud, timeout=0.2)
        except (serial.SerialException, OSError) as e:
            raise ArmError(f"{port} 를 열 수 없습니다: {e}") from e

        # 포트를 열면 보드가 재시작되고 "READY" 를 보낸다
        if self._wait_for("READY", self.ready_timeout):
            self._log(f"아두이노 연결됨 ({port}, READY 수신)")
        else:
            self._log(f"경고: {port} 에서 READY 를 받지 못했습니다. 재시작되지 않는 보드이면 정상일 수 있어서 계속 진행합니다.")
        self._ser.reset_input_buffer()
        return port

    def close(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    @property
    def busy(self):
        """팔이 동작 중이면 True"""
        return self._busy_lock.locked()

    def trigger(self, wait=True):
        """팔을 한 번 동작시킨다.
        wait=True  : DONE 을 받을 때까지 기다림. DONE 을 받으면 True, 시간 초과면 False
        wait=False : 명령만 보내고 바로 돌아옴. 보냈으면 True
        이미 동작 중이면 명령을 보내지 않고 False (명령이 쌓여서 연달아 움직이는 걸 막음)
        """
        if self._ser is None:
            raise ArmError("연결되지 않았습니다. 먼저 connect() 를 호출하세요.")
        if not self._busy_lock.acquire(blocking=False):
            return False
        try:
            self._send()
        except Exception:
            self._busy_lock.release()
            raise

        if wait:
            try:
                return self._wait_done()
            finally:
                self._busy_lock.release()

        def background():
            try:
                self._wait_done()
            finally:
                self._busy_lock.release()

        threading.Thread(target=background, daemon=True).start()
        return True

    def _send(self):
        for attempt in (1, 2):
            try:
                self._ser.reset_input_buffer()    # 이전에 남은 응답은 버림
                self._ser.write(b"T")
                self._ser.flush()
                return
            except (serial.SerialException, OSError) as e:
                if attempt == 2:
                    raise ArmError(f"명령을 보내지 못했습니다: {e}") from e
                self._log(f"시리얼 오류, 재연결을 시도합니다: {e}")
                self.close()
                self.connect()

    def _wait_for(self, token, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self._ser.readline().decode(errors="ignore").strip()
            except (serial.SerialException, OSError) as e:
                raise ArmError(f"읽는 중 시리얼 오류: {e}") from e
            if token in line:
                return True
        return False

    def _wait_done(self):
        if self._wait_for("DONE", self.done_timeout):
            return True
        self._log(f"경고: {self.done_timeout:.0f}초 안에 DONE 을 받지 못했습니다. 팔 전원과 배선을 확인하세요.")
        return False


class ArmTrigger:
    """추론 서버가 보내는 탐지 결과를 보고, 언제 팔을 움직일지 정한다.
    - 쓰레기가 confirm 프레임 연속으로 탐지돼야 움직임 (한 프레임짜리 오탐으로 움직이지 않게)
    - 팔이 동작 중이거나, 동작이 끝난 뒤 cooldown 초 동안은 탐지를 무시하고 연속 횟수를 0 으로 되돌림
      (같은 쓰레기로 팔이 여러 번 움직이지 않게)
    """

    def __init__(self, arm, confirm=ARM_CONFIRM, cooldown=ARM_COOLDOWN, classes=None, log=print,
                 delay=ARM_DELAY):
        self.arm = arm
        self.delay = max(0.0, float(delay))
        self.confirm = max(1, int(confirm))
        self.cooldown = max(0.0, float(cooldown))
        self.classes = set(classes) if classes else None
        self._log = log
        self._lock = threading.Lock()
        self._hits = 0
        self._running = False
        self._last_done = 0.0
        self.count = 0

    def on_detections(self, dets):
        if self.classes is not None:
            dets = [d for d in dets if d.get("cls") in self.classes]
        with self._lock:
            if self._running or time.time() - self._last_done < self.cooldown:
                self._hits = 0
                return
            if not dets:
                self._hits = 0
                return
            self._hits += 1
            if self._hits < self.confirm:
                return
            self._hits = 0
            self._running = True
            self.count += 1
            n = self.count
        names = ", ".join(f"{d.get('cls')} {float(d.get('conf', 0)):.2f}" for d in dets[:3])
        wait = f" ({self.delay:.1f}초 후)" if self.delay > 0 else ""
        self._log(f"[분류 팔 #{n}] 쓰레기 감지 ({names}) -> 팔 동작{wait}")
        threading.Thread(target=self._run, args=(n,), daemon=True).start()

    def _run(self, n):
        if self.delay > 0:
            time.sleep(self.delay)   # 기다리는 동안에도 _running 이라 새 탐지는 무시됨
        t0 = time.time()
        try:
            if self.arm.trigger(wait=True):
                self._log(f"[분류 팔 #{n}] 동작 완료 ({time.time() - t0:.1f}초)")
        except ArmError as e:
            self._log(f"[분류 팔 #{n}] 오류: {e}")
        except Exception as e:
            self._log(f"[분류 팔 #{n}] 예상치 못한 오류: {e}")
        finally:
            with self._lock:
                self._running = False
                self._last_done = time.time()


ARM_TRIGGER = None   # main() 에서 팔이 연결되면 ArmTrigger 가 들어감


@sio.on("detections")
def on_detections(dets):
    if ARM_TRIGGER is None:
        return
    try:
        ARM_TRIGGER.on_detections(dets or [])
    except Exception as e:
        print(f"탐지 결과 처리 오류: {e}")


def get_host_ip():
    try:
        out = subprocess.check_output(["hostname", "-I"]).decode().split()
        return out[0] if out else "<Jetson IP>"
    except Exception:
        return "<Jetson IP>"


def start_container(conf=DEFAULT_CONF):
    with open(SERVER_FILE, "w", encoding="utf-8") as f:
        f.write(SERVER_CODE)

    print("docker 실행을 위해 sudo 비밀번호가 필요할 수 있습니다.")
    subprocess.run(["sudo", "-v"], check=False)

    # 이전에 남아있는 같은 이름의 컨테이너 정리
    subprocess.run(["sudo", "docker", "rm", "-f", CONTAINER_NAME],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cmd = [
        "sudo", "docker", "run", "--rm",
        "--runtime", "nvidia", "--network", "host",
        "--name", CONTAINER_NAME,
        "-e", "PYTHONUNBUFFERED=1",
        "-e", f"CONF_THRESHOLD={conf}",
        "-v", f"{HOME}:/workspace",
        "-w", "/workspace",
        DOCKER_IMAGE,
        "python3", "/workspace/_yolo_server.py",
    ]
    print("추론 서버 컨테이너 시작 중...")
    return subprocess.Popen(cmd)


def wait_for_server(container_proc, timeout=180):
    print("모델 로딩 대기 중... (TensorRT 엔진 로딩에 수십 초 걸릴 수 있어요)")
    start = time.time()
    while time.time() - start < timeout:
        if container_proc.poll() is not None:
            print("컨테이너가 종료되었습니다. 위 에러 로그를 확인하세요.")
            return False
        try:
            with socket.create_connection(("localhost", SERVER_PORT), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    print("서버가 시간 안에 시작되지 않았습니다.")
    return False


def read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ── 카메라 선택 (CSI / USB) ───────────────────────────
def list_video_nodes():
    """/sys/class/video4linux 를 읽어서 비디오 장치 목록을 만든다."""
    def read(path):
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        except OSError:
            return ""

    nodes = []
    for path in glob.glob("/sys/class/video4linux/video*"):
        base = os.path.basename(path)
        nodes.append({
            "dev": f"/dev/{base}",
            "num": int(re.sub(r"\D", "", base) or 0),
            "name": read(os.path.join(path, "name")),
            "index": read(os.path.join(path, "index")),
        })
    return sorted(nodes, key=lambda n: n["num"])


def is_csi_node(node):
    name = node["name"].lower()
    return "vi-output" in name or "imx" in name or "tegra" in name


def pick_camera(camera, device, nodes):
    """(모드, USB 후보 장치 목록) 반환. 모드: 'csi' / 'usb' / None(카메라 없음)."""
    if device is not None:
        dev = f"/dev/video{device}" if str(device).isdigit() else str(device)
        return "usb", [dev]

    # USB 카메라는 장치 파일이 2개(영상용 index 0, 메타데이터용 index 1)라서 영상용만 후보로 삼는다
    usb = [n["dev"] for n in nodes if not is_csi_node(n) and n["index"] in ("", "0")]
    has_csi = any(is_csi_node(n) for n in nodes)

    if camera == "csi":
        return "csi", []
    if camera == "usb":
        return ("usb", usb) if usb else (None, [])
    if has_csi:
        return "csi", []
    if usb:
        return "usb", usb
    return None, []


def fit_frame(frame):
    """어떤 해상도/비율로 들어와도 OUT_W x OUT_H(4:3) 로 맞춘다 (가운데를 잘라서 비율 유지)."""
    h, w = frame.shape[:2]
    if (w, h) == (OUT_W, OUT_H):
        return frame
    target = OUT_W / OUT_H
    if w / h > target:
        new_w = int(h * target)
        x0 = (w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    elif w / h < target:
        new_h = int(w / target)
        y0 = (h - new_h) // 2
        frame = frame[y0:y0 + new_h, :]
    return cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)


_CV_FLIP = {2: -1, 4: 1, 6: 0}  # gst 의 flip-method 번호 -> cv2.flip 코드 (180도 / 좌우 / 상하)


def apply_flip(frame, flip):
    code = _CV_FLIP.get(flip)
    return frame if code is None else cv2.flip(frame, code)


class CsiSource:
    """CSI 카메라: gst-launch 로 프레임을 읽는다."""
    label = "CSI 카메라"

    def __init__(self, flip):
        self.flip = flip
        self.proc = None

    def open(self):
        self.proc = subprocess.Popen(build_gst_cmd(self.flip), stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        return True

    def read(self):
        raw = read_exact(self.proc.stdout, FRAME_BYTES)
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape((OUT_H, OUT_W, 3))

    def error_text(self):
        try:
            return self.proc.stderr.read().decode(errors="ignore")
        except Exception:
            return ""

    def close(self):
        if self.proc is not None:
            self.proc.terminate()


def _open_cv_capture(dev):
    """OpenCV 로 USB 카메라를 연다.
    최근 OpenCV 는 V4L2 를 '/dev/video0' 같은 경로 문자열로는 못 열고
    ('can't be used to capture by name') 번호로만 열 수 있어서, 번호로 먼저 시도한다."""
    m = re.fullmatch(r"/dev/video(\d+)", str(dev))
    attempts = []
    if m:
        attempts += [(int(m.group(1)), cv2.CAP_V4L2), (int(m.group(1)), cv2.CAP_ANY)]
    attempts.append((dev, cv2.CAP_V4L2))
    for target, api in attempts:
        cap = cv2.VideoCapture(target, api)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def build_usb_gst_cmds(dev):
    """OpenCV 로 안 열릴 때 쓰는 gst-launch 파이프라인들 (순서대로 시도)."""
    caps_out = f"video/x-raw,format=BGR,width={OUT_W},height={OUT_H}"
    head = ["gst-launch-1.0", "-q", "v4l2src", f"device={dev}", "!"]
    tail = ["!", "videoconvert", "!", "videoscale", "!", caps_out, "!", "fdsink", "fd=1"]
    return [
        head + [f"image/jpeg,width={OUT_W},height={OUT_H},framerate=30/1", "!", "jpegdec"] + tail,  # MJPG
        head + [f"video/x-raw,width={OUT_W},height={OUT_H}"] + tail,                                 # 원시(YUYV)
        head + ["decodebin"] + tail,                                                                  # 자동
    ]


def read_first_frame(proc, timeout=6.0):
    """gst 프로세스에서 첫 프레임을 받는다. 시간 안에 안 오거나 프로세스가 끝나면 None."""
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        return None
    return read_exact(proc.stdout, FRAME_BYTES)


class UsbSource:
    """USB 카메라(UVC 웹캠): OpenCV(V4L2)로 읽고, 안 되면 gst-launch 로 읽는다."""
    label = "USB 카메라"

    def __init__(self, devices, flip):
        self.devices = devices
        self.flip = flip
        self.cap = None
        self.proc = None
        self._first = None
        self.err = ""
        self.gst_errors = []
        if flip not in (0, 2, 4, 6):
            print(f"USB 카메라는 뒤집기 값 0/2/4/6 만 지원합니다 (입력값 {flip}). 뒤집지 않고 진행합니다.")
            self.flip = 0

    def _try_cv(self, dev):
        # 먼저 MJPG(대부분의 웹캠에서 640x480 30fps 가능)로, 안 되면 기본 형식으로 시도
        for use_mjpg in (True, False):
            cap = _open_cv_capture(dev)
            if cap is None:
                return None
            if use_mjpg:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, OUT_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OUT_H)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            cap.release()
        return None

    def _try_gst(self, dev):
        for cmd in build_usb_gst_cmds(dev):
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            except OSError as e:
                self.gst_errors.append(f"gst-launch-1.0 를 실행하지 못했습니다: {e}")
                return False
            raw = read_first_frame(proc)
            if raw is not None:
                self.proc = proc
                self._first = np.frombuffer(raw, dtype=np.uint8).reshape((OUT_H, OUT_W, 3))
                return True
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
            try:
                text = proc.stderr.read().decode(errors="ignore").strip()
                if text:
                    self.gst_errors.append(text.splitlines()[-1])
            except Exception:
                pass
        return False

    def open(self):
        tried = []
        for dev in self.devices:
            cap = self._try_cv(dev)
            if cap is not None:
                self.cap = cap
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"USB 카메라 열림: {dev} (OpenCV, {w}x{h})")
                return True
            if self._try_gst(dev):
                print(f"USB 카메라 열림: {dev} (GStreamer, {OUT_W}x{OUT_H})")
                return True
            tried.append(dev)
        self.err = ("USB 카메라를 열지 못했습니다. 시도한 장치: " + (", ".join(tried) or "없음") + "\n"
                    "   다른 프로그램이 카메라를 쓰고 있지 않은지, `ls /dev/video*` 로 장치가 보이는지 확인하세요.")
        if self.gst_errors:
            self.err += "\n   GStreamer 오류: " + " | ".join(self.gst_errors[-3:])
        return False

    def read(self):
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return None
            frame = fit_frame(frame)
        elif self._first is not None:
            frame, self._first = self._first, None
        else:
            raw = read_exact(self.proc.stdout, FRAME_BYTES)
            if raw is None:
                return None
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((OUT_H, OUT_W, 3))
        return np.ascontiguousarray(apply_flip(frame, self.flip))

    def error_text(self):
        if self.err:
            return self.err
        if self.proc is not None:
            try:
                return self.proc.stderr.read().decode(errors="ignore")
            except Exception:
                return ""
        return ""

    def close(self):
        if self.cap is not None:
            self.cap.release()
        if self.proc is not None:
            self.proc.terminate()


GAIN = None  # 프레임에 곱할 보정 지도 (H x W x 3, BGR 순서). 없으면 보정 안 함


def load_gain():
    if not os.path.exists(GAIN_FILE):
        print("보정 파일이 없어서 색 보정 없이 실행합니다. (보정하려면 --calibrate)")
        return None
    gain = np.load(GAIN_FILE)
    if gain.shape != (OUT_H, OUT_W, 3):
        print(f"보정 파일 크기({gain.shape})가 현재 해상도({OUT_H}, {OUT_W}, 3)와 달라 무시합니다. 다시 --calibrate 하세요.")
        return None
    print("렌즈 쉐이딩 보정 적용 중")
    return gain.astype(np.float32)


def apply_shading(frame):
    if GAIN is None:
        return frame
    return np.clip(frame.astype(np.float32) * GAIN, 0, 255).astype(np.uint8)


LENS_MAPS = None   # (map1, map2) 렌즈 왜곡을 펴는 변환 지도. 없으면 적용 안 함
LATEST_RAW = None  # 렌즈 보정용 체스보드 탐색에 쓰는 가장 최근 원본 프레임


def load_lens():
    if not os.path.exists(LENS_FILE):
        print("렌즈 보정 파일이 없어서 왜곡 보정 없이 실행합니다. (보정하려면 --lens-calibrate)")
        return None
    data = np.load(LENS_FILE)
    map1, map2 = data["map1"], data["map2"]
    if map1.shape[:2] != (OUT_H, OUT_W):
        print("렌즈 보정 파일 크기가 현재 해상도와 달라 무시합니다. 다시 --lens-calibrate 하세요.")
        return None
    print("렌즈 왜곡 보정 적용 중")
    return (map1, map2)


def apply_lens(frame):
    if LENS_MAPS is None:
        return frame
    return cv2.remap(frame, LENS_MAPS[0], LENS_MAPS[1], cv2.INTER_LINEAR)


def lens_calibrate_worker():
    """카메라 앞에 보여주는 체스보드를 자동으로 수집하고, 충분히 모이면 렌즈 왜곡 보정 지도를 계산한다."""
    global LENS_MAPS

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    sub_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    collected = []

    print("\n" + "=" * 60)
    print(f"렌즈 보정 모드: 체스보드를 카메라 앞에 보여주세요 (목표 {LENS_VIEWS}장면)")
    print("   - 체스보드 전체(모든 모서리)가 화면 안에 보여야 인식됩니다")
    print("   - 화면 중앙/가장자리/모서리, 가까이/멀리, 여러 각도로 기울여가며 보여주세요")
    print("=" * 60 + "\n")

    while len(collected) < LENS_VIEWS:
        time.sleep(LENS_INTERVAL)
        frame = LATEST_RAW
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cb_flags)
        if not found:
            continue

        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), sub_criteria)
        pts = corners.reshape(-1, 2)

        # 이전에 수집한 장면과 거의 같으면 건너뜀 (같은 자리에 가만히 있는 경우)
        if any(np.abs(pts - p).mean() < LENS_MIN_MOVE for p in collected):
            continue

        collected.append(pts)
        print(f"체스보드 인식 ({len(collected)}/{LENS_VIEWS}) - 위치/각도를 바꿔서 계속 보여주세요")

    print("\n렌즈 보정 계산 중...")
    img_points = [p.reshape(1, -1, 2).astype(np.float64) for p in collected]
    obj_points = [objp] * len(img_points)

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    try:
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            obj_points, img_points, (OUT_W, OUT_H), K, D, None, None, flags, criteria)
    except cv2.error as e:
        print(f"렌즈 보정 계산 실패: {e}")
        print("   체스보드를 더 다양한 위치/각도로 보여주고 --lens-calibrate 를 다시 실행하세요.")
        return

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (OUT_W, OUT_H), np.eye(3), balance=LENS_BALANCE)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (OUT_W, OUT_H), cv2.CV_16SC2)

    np.savez(LENS_FILE, map1=map1, map2=map2, rms=rms)
    LENS_MAPS = (map1, map2)

    print(f"렌즈 보정 완료! 재투영 오차(RMS): {rms:.2f}px  (1px 안팎이면 좋음)")
    if rms > 2.0:
        print("오차가 큽니다. 체스보드를 더 다양한 위치/각도로 다시 수집하는 걸 권장합니다.")
    print(f"   저장: {LENS_FILE}")
    print("   지금부터 브라우저 화면에 바로 적용됩니다. 평소 실행 때도 자동으로 적용돼요.\n")


def build_lens_maps_from_fov(diag_fov_deg):
    """렌즈가 등거리 어안 렌즈라고 가정하고, 대각 시야각(도)으로 왜곡을 펴는 지도를 만든다."""
    half_diag = 0.5 * math.hypot(OUT_W, OUT_H)
    f = half_diag / math.radians(diag_fov_deg / 2.0)
    K = np.array([[f, 0.0, OUT_W / 2.0], [0.0, f, OUT_H / 2.0], [0.0, 0.0, 1.0]])
    D = np.zeros((4, 1))
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (OUT_W, OUT_H), np.eye(3), balance=LENS_BALANCE)
    return cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (OUT_W, OUT_H), cv2.CV_16SC2)


def lens_manual_watcher():
    """~/lens_fov.txt 의 시야각 숫자가 바뀔 때마다 왜곡 보정을 다시 만들어 바로 적용한다."""
    global LENS_MAPS

    if not os.path.exists(LENS_FOV_FILE):
        with open(LENS_FOV_FILE, "w") as f:
            f.write(str(LENS_FOV_DEFAULT))

    print("\n" + "=" * 60)
    print("렌즈 수동 보정 모드")
    print("   브라우저 화면을 보면서, 다른 터미널(호스트)에서 숫자를 바꿔보세요:")
    print("     echo 150 > ~/lens_fov.txt")
    print("   - 가장자리 직선이 아직 바깥으로 휘어 있으면 → 숫자를 올리세요 (예: 150, 160, 170)")
    print("   - 직선이 오히려 안쪽으로 휘면 → 숫자를 내리세요 (예: 130, 120)")
    print("   직선(벽 모서리, 천장 선 등)이 곧게 보이는 값이 정답이고, 자동으로 저장됩니다.")
    print("=" * 60 + "\n")

    last_mtime = None
    while True:
        try:
            mtime = os.path.getmtime(LENS_FOV_FILE)
            if mtime != last_mtime:
                last_mtime = mtime
                with open(LENS_FOV_FILE) as f:
                    fov = float(f.read().strip())
                if 60.0 <= fov <= 200.0:
                    map1, map2 = build_lens_maps_from_fov(fov)
                    LENS_MAPS = (map1, map2)
                    np.savez(LENS_FILE, map1=map1, map2=map2, rms=-1.0)
                    print(f"렌즈 시야각 {fov:.0f}° 로 왜곡 보정 적용 (저장됨)")
                else:
                    print("시야각은 60~200 사이 숫자로 입력하세요.")
        except Exception as e:
            print(f"시야각 파일을 읽지 못했습니다: {e}")
        time.sleep(1.0)


def calibrate_shading():
    """흰 종이(균일한 흰색 면)를 촬영해서 위치별 색/밝기 보정 지도를 만든다."""
    print("보정용 촬영 시작... (렌즈에 흰 종이를 덮은 상태인지 확인하세요)")
    gst = subprocess.Popen(GST_CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    acc = np.zeros((OUT_H, OUT_W, 3), dtype=np.float32)
    n = 0
    try:
        for i in range(CALIB_SKIP + CALIB_FRAMES):
            raw = read_exact(gst.stdout, FRAME_BYTES)
            if raw is None:
                print("카메라 스트림이 끝났습니다.")
                print(gst.stderr.read().decode(errors="ignore"))
                return
            if i >= CALIB_SKIP:
                acc += np.frombuffer(raw, dtype=np.uint8).reshape((OUT_H, OUT_W, 3)).astype(np.float32)
                n += 1
    finally:
        gst.terminate()

    mean = acc / n

    avg_level = float(mean.mean())
    print(f"평균 밝기: {avg_level:.0f} / 255")
    if avg_level < 40:
        print("너무 어둡습니다. 더 밝은 쪽을 향하게 하거나 종이를 한 겹 줄여서 다시 하세요.")
    if avg_level > 235:
        print("너무 밝아서 포화됐을 수 있습니다. 종이를 한 겹 늘리거나 조명을 줄여서 다시 하세요.")

    # 먼지/노이즈가 지도에 들어가지 않게 작게 줄였다가 부드럽게 키움
    small = cv2.resize(mean, (40, 30), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    smooth = cv2.resize(small, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)
    smooth = np.maximum(smooth, 1.0)

    B, G, R = smooth[:, :, 0], smooth[:, :, 1], smooth[:, :, 2]

    # 색 보정: 모든 위치에서 R, B 를 G 와 같은 수준으로 맞춤 (회색이 회색으로 보이게)
    gain_r = G / R
    gain_b = G / B
    color_dev = float(max(np.abs(R / G - 1).max(), np.abs(B / G - 1).max()))

    # 밝기 보정: 가장자리가 어두워지는 정도를 일부 보상
    r = OUT_H // 10
    ch, cw = OUT_H // 2, OUT_W // 2
    g_center = float(G[ch - r:ch + r, cw - r:cw + r].mean())
    gain_l = np.clip((g_center / G) ** LUMA_STRENGTH, 1.0, 3.0)

    gain = np.dstack([gain_b * gain_l, gain_l, gain_r * gain_l]).astype(np.float32)
    gain = np.clip(gain, 0.5, 4.0)
    np.save(GAIN_FILE, gain)

    print(f"위치별 최대 색 편차(보정 전): {color_dev * 100:.0f}%")
    print(f"보정 배율 범위 - B: {gain[:, :, 0].min():.2f}~{gain[:, :, 0].max():.2f}, "
          f"G: {gain[:, :, 1].min():.2f}~{gain[:, :, 1].max():.2f}, "
          f"R: {gain[:, :, 2].min():.2f}~{gain[:, :, 2].max():.2f}")
    print(f"보정 지도 저장 완료: {GAIN_FILE}")
    print("   이제 종이를 치우고 python3 ai_camera.py 를 실행하세요.")


_stats = {"read": 0, "sent": 0}   # 5초마다 출력하는 속도 측정용 카운터


def stats_printer(interval=5.0):
    """카메라에서 읽는 속도와 서버로 보내는 속도를 주기적으로 출력 (어디서 느려지는지 확인용)."""
    last_read = last_sent = 0
    last_t = time.time()
    while True:
        time.sleep(interval)
        now = time.time()
        dt = now - last_t
        r, s_ = _stats["read"], _stats["sent"]
        print(f"카메라에서 {(r - last_read) / dt:.1f}장/초 읽음 | 서버로 {(s_ - last_sent) / dt:.1f}장/초 전송", flush=True)
        last_read, last_sent, last_t = r, s_, now


def send_frames():
    last_send = 0.0
    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        now = time.time()
        if now - last_send < SEND_INTERVAL:
            continue
        last_send = now
        try:
            frame = apply_shading(frame)   # 색 보정은 렌즈 원본 좌표 기준이므로 먼저
            frame = apply_lens(frame)      # 그다음 렌즈 왜곡을 펴줌
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            sio.emit("send_frame", base64.b64encode(buffer).decode())
            _stats["sent"] += 1
        except Exception as e:
            print(f"전송 오류: {e}")


def arm_test(port):
    """카메라/서버 없이 팔만 한 번 움직여 보고 걸린 시간을 출력한다."""
    arm = ArmController(port)
    try:
        arm.connect()
        t0 = time.time()
        ok = arm.trigger(wait=True)
        print(f"{'DONE 수신' if ok else '시간 초과'} ({time.time() - t0:.1f}초)")
    except ArmError as e:
        print(f"오류: {e}")
        sys.exit(1)
    finally:
        arm.close()


def main():
    global GAIN, LENS_MAPS, LATEST_RAW, LENS_FILE, GST_CMD, SEND_INTERVAL, ARM_TRIGGER

    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-calibrate", action="store_true",
                        help="평소처럼 실행하면서 체스보드를 수집해 렌즈 왜곡 보정 지도를 만듦")
    parser.add_argument("--lens-manual", action="store_true",
                        help="체스보드 없이, ~/lens_fov.txt 의 시야각 숫자를 바꿔가며 왜곡 보정을 눈으로 맞춤")
    parser.add_argument("--no-lens", action="store_true", help="렌즈 왜곡 보정을 적용하지 않고 실행")
    parser.add_argument("--calibrate", action="store_true",
                        help="흰 종이를 촬영해서 렌즈 쉐이딩 보정 지도를 만들고 종료 (CSI 전용)")
    parser.add_argument("--no-shading", action="store_true", help="색 보정을 적용하지 않고 실행")
    parser.add_argument("--camera", choices=["auto", "csi", "usb"], default="auto",
                        help="카메라 종류 (기본 auto: CSI 가 보이면 CSI, 아니면 USB)")
    parser.add_argument("--device", default=None,
                        help="USB 카메라 장치 번호나 경로 (예: 0 또는 /dev/video0). 주면 USB 모드")
    parser.add_argument("--flip", type=int, default=None,
                        help="화면 뒤집기 (0 없음, 2 180도, 4 좌우, 6 상하). 기본: CSI 는 FLIP_METHOD 설정값, USB 는 0")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"탐지 신뢰도 기준 (기본 {DEFAULT_CONF}). 낮추면 더 많이 잡히지만 오탐도 늘어요 (예: 0.3)")
    parser.add_argument("--send-fps", type=float, default=None,
                        help="서버로 보내는 최대 초당 장수 (기본 20). 예: 30")
    # 분류 팔
    parser.add_argument("--no-arm", action="store_true", help="분류 팔(아두이노) 없이 실행")
    parser.add_argument("--arm-port", default=None,
                        help="아두이노 시리얼 포트 (예: /dev/ttyACM0). 없으면 자동 감지")
    parser.add_argument("--arm-confirm", type=int, default=ARM_CONFIRM,
                        help=f"이만큼 연속 프레임에서 탐지돼야 팔을 움직임 (기본 {ARM_CONFIRM})")
    parser.add_argument("--arm-cooldown", type=float, default=ARM_COOLDOWN,
                        help=f"팔 동작이 끝난 뒤 쉬는 시간(초) (기본 {ARM_COOLDOWN})")
    parser.add_argument("--arm-delay", type=float, default=ARM_DELAY,
                        help=f"탐지가 확정된 뒤 팔을 움직이기 전까지 기다리는 시간(초) (기본 {ARM_DELAY}, 바로 동작)")
    parser.add_argument("--arm-classes", default=None,
                        help="이 클래스가 탐지될 때만 팔을 움직임. 쉼표로 여러 개 (예: 일반쓰레기,종이). 기본: 전체")
    parser.add_argument("--arm-test", action="store_true",
                        help="카메라/서버 없이 팔을 한 번 움직여 보고 종료")
    parser.add_argument("--list-ports", action="store_true", help="시리얼 포트 목록만 출력하고 종료")
    args = parser.parse_args()

    if args.list_ports:
        print_serial_ports()
        return
    if args.arm_test:
        arm_test(args.arm_port)
        return

    if args.send_fps:
        SEND_INTERVAL = 1.0 / args.send_fps

    mode, usb_devices = pick_camera(args.camera, args.device, list_video_nodes())
    if mode is None:
        print("카메라를 찾지 못했습니다. (/dev/video* 장치가 없어요)")
        print("   - USB 카메라가 꽂혀 있는지, `ls /dev/video*` 로 보이는지 확인하세요.")
        sys.exit(1)

    flip = args.flip if args.flip is not None else (FLIP_METHOD if mode == "csi" else 0)

    if mode == "csi":
        GST_CMD = build_gst_cmd(flip)
        print("카메라 모드: CSI")
    else:
        # CSI(imx219) 카메라용으로 만든 보정 파일은 USB 카메라에 맞지 않으므로 쓰지 않는다.
        LENS_FILE = os.path.join(HOME, "lens_maps_usb.npz")
        print(f"카메라 모드: USB ({', '.join(usb_devices)})")
        print("   CSI(imx219)용 색 번짐/렌즈 보정은 USB 카메라에 맞지 않아 적용하지 않습니다.")

    if args.calibrate:
        if mode != "csi":
            print("--calibrate(색 번짐 보정)는 CSI(imx219) 카메라 전용입니다.")
            return
        calibrate_shading()
        return

    if mode == "csi" and not args.no_shading:
        GAIN = load_gain()

    # 렌즈 보정 수집 중에는 기존 보정 없이 시작 (수집이 끝나면 새 보정이 자동 적용됨)
    if not args.no_lens and not args.lens_calibrate and not args.lens_manual:
        LENS_MAPS = load_lens()

    if not os.path.exists(ENGINE_FILE):
        print(f"{ENGINE_FILE} 파일이 없습니다. best.engine 을 홈 폴더에 두세요.")
        sys.exit(1)

    # 분류 팔 연결 (실패해도 카메라/웹 화면은 그대로 실행)
    arm = None
    if args.no_arm:
        print("분류 팔: 사용 안 함 (--no-arm)")
    elif args.lens_calibrate or args.lens_manual:
        print("분류 팔: 렌즈 보정 모드에서는 사용하지 않습니다.")
    else:
        arm = ArmController(args.arm_port)
        try:
            arm.connect()
            classes = [c.strip() for c in args.arm_classes.split(",") if c.strip()] if args.arm_classes else None
            ARM_TRIGGER = ArmTrigger(arm, args.arm_confirm, args.arm_cooldown, classes, delay=args.arm_delay)
            target = ", ".join(classes) if classes else "모든 클래스"
            print(f"분류 팔: {target} 가 {ARM_TRIGGER.confirm}프레임 연속 탐지되면 "
                  f"{ARM_TRIGGER.delay:.1f}초 후 동작, 동작 후 {ARM_TRIGGER.cooldown:.1f}초 쉼")
        except ArmError as e:
            print(f"분류 팔 없이 실행합니다: {e}")
            arm.close()
            arm = None

    source = CsiSource(flip) if mode == "csi" else UsbSource(usb_devices, flip)
    print(f"신뢰도 기준 conf={args.conf}, 서버로 최대 {1.0 / SEND_INTERVAL:.0f}장/초 전송, 전송 화질 {JPEG_QUALITY}")
    container_proc = start_container(args.conf)

    try:
        if not wait_for_server(container_proc):
            return

        sio.connect(f"http://localhost:{SERVER_PORT}")
        threading.Thread(target=send_frames, daemon=True).start()

        print(f"{source.label} 시작...")
        if not source.open():
            print(source.error_text())
            return

        if args.lens_calibrate:
            threading.Thread(target=lens_calibrate_worker, daemon=True).start()
        if args.lens_manual:
            threading.Thread(target=lens_manual_watcher, daemon=True).start()
        threading.Thread(target=stats_printer, daemon=True).start()

        first = True
        while True:
            frame = source.read()
            if frame is None:
                print("카메라 스트림이 끝났습니다.")
                err = source.error_text()
                if err:
                    print(err)
                if mode == "csi":
                    print("   CSI 카메라가 안 열리면 USB 카메라로: python3 ~/ai_camera.py --camera usb")
                break

            if first:
                print("\n" + "=" * 60)
                print("카메라 연결 완료! 스트리밍 중 (Ctrl+C 로 종료)")
                print(f"다른 노트북 브라우저에서 접속: http://{get_host_ip()}:{SERVER_PORT}")
                print("=" * 60 + "\n")
                first = False

            LATEST_RAW = frame   # 렌즈 보정 수집 스레드가 사용 (보정 전 원본)
            _stats["read"] += 1
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                    frame_queue.put_nowait(frame)
                except queue.Empty:
                    pass

    except KeyboardInterrupt:
        print("\n종료합니다...")
    finally:
        ARM_TRIGGER = None
        source.close()
        try:
            sio.disconnect()
        except Exception:
            pass
        if arm is not None:
            arm.close()
        subprocess.run(["sudo", "docker", "stop", CONTAINER_NAME],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("정리 완료")


if __name__ == "__main__":
    main()