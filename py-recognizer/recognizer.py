# recognizer.py
import cv2
import threading
import time
from deepface import DeepFace
import numpy as np
import os
import requests
import datetime

class FaceRecognizer:
    def __init__(self, faces_dir="faces", embed_dir="embeddings"):
        self.faces_dir = faces_dir
        self.embed_dir = embed_dir
        self.gallery = {}
        self.last_result = {"recognized": False, "name": None, "confidence": 0.0}
        self._load_gallery()

    def _load_gallery(self):
        """加载人脸图库"""
        for file in os.listdir(self.faces_dir):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(self.faces_dir, file)
            name = os.path.splitext(file)[0]
            self.gallery[name] = path
            print(f"[INFO] 已加载人脸样本: {file}")

    def recognize(self, frame):
        """识别单帧"""
        try:
            result = DeepFace.find(
                img_path=frame,
                db_path=self.faces_dir,
                model_name="Facenet",
                enforce_detection=False,
                silent=True
            )

            if len(result) > 0 and len(result[0]) > 0:
                best = result[0].iloc[0]
                name = os.path.basename(os.path.dirname(best["identity"]))
                confidence = 1 - float(best["distance"])  # 相似度反向化
                self.last_result = {"recognized": True, "name": name, "confidence": confidence}
                print(f"[{time.strftime('%H:%M:%S')}] 识别到 {name} ({confidence:.3f})")

                # ✅ 新增：识别成功后汇报到 Go 后端
                post_record_to_go(name, confidence)
            else:
                self.last_result = {"recognized": False, "name": None, "confidence": 0.0}

        except Exception as e:
            print(f"[WARN] 识别异常: {e}")
            self.last_result = {"recognized": False, "name": None, "confidence": 0.0}

        return self.last_result

    def get_last_result(self):
        return self.last_result


# ✅ 新增函数：回传识别记录到 Go 后端
def post_record_to_go(name, confidence):
    """把识别结果回传到 Go 服务"""
    try:
        payload = {
            "name": name,
            "confidence": float(confidence),
            "timestamp": datetime.datetime.now().isoformat()
        }
        # Go 后端监听的端口
        requests.post("http://127.0.0.1:8080/api/records", json=payload, timeout=1.0)
    except Exception as e:
        # 不影响主识别流程
        print(f"[WARN] 回传失败: {e}")
