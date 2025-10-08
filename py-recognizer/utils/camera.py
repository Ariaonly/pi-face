import cv2
import threading
import time
from datetime import datetime

class CameraThread:
    def __init__(self, recognizer, interval=2.0):
        self.recognizer = recognizer
        self.interval = interval     # 每隔多少秒识别一次
        self.running = False
        self.thread = None

    def _loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] 摄像头打开失败")
            return
        print("[INFO] 摄像头已启动")
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.2)
                continue

            # OpenCV 默认是 BGR → 转为 RGB
            rgb = frame[:, :, ::-1]
            res = self.recognizer.recognize(rgb)
            if res["recognized"]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 识别到 {res['name']} ({res['confidence']:.3f})")
            time.sleep(self.interval)
        cap.release()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

