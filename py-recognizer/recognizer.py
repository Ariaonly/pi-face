import cv2
import numpy as np
import os
from deepface import DeepFace

class FaceRecognizer:
    def __init__(self, faces_dir="faces", embed_dir="embeddings"):
        self.faces_dir = faces_dir
        self.embed_dir = embed_dir
        self.known_faces = {}
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.load_faces()

    def load_faces(self):
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
        for fn in os.listdir(self.faces_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(self.faces_dir, fn)
            self.known_faces[os.path.splitext(fn)[0]] = path
            print(f"[INFO] 已加载人脸样本: {fn}")

    def recognize(self, frame_rgb):
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return {"recognized": False, "name": "NoFace", "confidence": 0.0}

        (x, y, w, h) = faces[0]
        face = frame_rgb[y:y+h, x:x+w]

        best_match = {"name": "Unknown", "distance": 1.0}
        for name, path in self.known_faces.items():
            try:
                result = DeepFace.verify(face, path, model_name="Facenet", enforce_detection=False)
                if result["verified"] and result["distance"] < best_match["distance"]:
                    best_match = {"name": name, "distance": result["distance"]}
            except Exception as e:
                print(f"[WARN] 比对 {name} 出错: {e}")

        recognized = best_match["name"] != "Unknown"
        confidence = round(1 - best_match["distance"], 3) if recognized else 0.0
        return {
            "recognized": recognized,
            "name": best_match["name"],
            "confidence": confidence
        }
