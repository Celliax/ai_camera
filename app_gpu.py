from flask import Flask
from flask_socketio import SocketIO
from ultralytics import YOLO
import cv2
import base64
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# GPU 모델 로드 - v2 (병/플라스틱/종이/일반쓰레기) 모델로 변경
model = YOLO(r"D:\waste-dataset\runs\v2_4class\weights\best.pt")
print("✅ GPU 모델 로드 완료! (v2: 병/플라스틱/종이/일반쓰레기)")

CAM_INDEX = 0  # 웹캠이 여러 개면 1, 2 등으로 변경

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>실시간 쓰레기 탐지 (GPU)</title>
        <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
        <style>
            body { font-family: Arial; text-align: center; padding: 20px; background: #f0f0f0; }
            .container { background: white; padding: 30px; border-radius: 10px; max-width: 800px; margin: 0 auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            img { max-width: 100%; border: 2px solid #333; border-radius: 5px; }
            #status { font-size: 18px; font-weight: bold; padding: 10px; }
            .connected { color: green; }
            .disconnected { color: red; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 실시간 쓰레기 탐지 (웹캠 + GPU, v2)</h1>
            <p id="status" class="disconnected">연결 중...</p>
            <img id="frame" src="" style="width: 100%;">
        </div>

        <script>
            const socket = io();

            socket.on('connect', () => {
                document.getElementById('status').innerHTML = '✅ 연결됨';
                document.getElementById('status').className = 'connected';
            });

            socket.on('frame', (data) => {
                document.getElementById('frame').src = 'data:image/jpeg;base64,' + data;
            });

            socket.on('disconnect', () => {
                document.getElementById('status').innerHTML = '❌ 연결 끊김';
                document.getElementById('status').className = 'disconnected';
            });
        </script>
    </body>
    </html>
    '''


def capture_and_infer():
    """서버 자체 웹캠을 열어서 추론 후 브라우저로 계속 브로드캐스트"""
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다. CAM_INDEX를 확인하세요.")
        return

    print("✅ 웹캠 준비 완료! 스트리밍 시작...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임 캡처 실패")
            time.sleep(0.1)
            continue

        results = model.predict(frame, conf=0.3, verbose=False)
        annotated = results[0].plot()

        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
        frame_b64 = base64.b64encode(buffer).decode()

        socketio.emit('frame', frame_b64)
        socketio.sleep(0.03)  # 약 30 FPS 상한


if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("\n" + "="*60)
    print("🚀 Windows GPU 서버 시작! (서버 자체 웹캠 사용, v2 모델)")
    print("="*60)
    print(f"\n🌐 접속 주소: http://localhost:5000")
    print("="*60 + "\n")

    # 웹캠 캡처를 백그라운드 스레드로 시작
    threading.Thread(target=capture_and_infer, daemon=True).start()

    socketio.run(app, host='0.0.0.0', port=5000, debug=False)