import face_recognition
import numpy as np
import os

class FaceRecognizer:
    def __init__(self, faces_dir="faces", embed_dir="embeddings"):
        self.faces_dir = faces_dir
        self.embed_dir = embed_dir
        self.known_encodings = []
        self.known_names = []
        self.load_faces()

    def load_faces(self):
        """从 faces/ 目录加载已知人脸并提取特征"""
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
        for name in os.listdir(self.faces_dir):
            path = os.path.join(self.faces_dir, name)
            if not name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            try:
                img = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(img)
                if len(encodings) > 0:
                    self.known_encodings.append(encodings[0])
                    base = os.path.splitext(name)[0]
                    self.known_names.append(base)
                    print(f"[INFO] 已加载人脸样本: {base}")
            except Exception as e:
                print(f"[WARN] 加载失败 {name}: {e}")

    def recognize(self, frame_rgb):
        """识别输入图像中的人脸，返回最匹配者"""
        face_locations = face_recognition.face_locations(frame_rgb)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        if len(face_encodings) == 0:
            return {"recognized": False, "name": "NoFace", "confidence": 0.0}

        face_enc = face_encodings[0]

        if not self.known_encodings:
            return {"recognized": False, "name": "Unknown", "confidence": 0.0}

        # 比较
        distances = face_recognition.face_distance(self.known_encodings, face_enc)
        best_idx = np.argmin(distances)
        best_dist = distances[best_idx]
        name = self.known_names[best_idx] if best_dist < 0.45 else "Unknown"

        return {
            "recognized": name != "Unknown",
            "name": name,
            "confidence": round(float(1 - best_dist), 3),
        }

