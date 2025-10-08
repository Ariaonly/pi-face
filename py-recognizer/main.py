from flask import Flask, request, jsonify
from datetime import datetime
from PIL import Image
import base64, io, numpy as np
from recognizer import FaceRecognizer
from utils.camera import CameraThread

app = Flask(__name__)
recog = FaceRecognizer(faces_dir="faces", embed_dir="embeddings")

# 启动摄像头线程
camera = CameraThread(recog, interval=3.0)
camera.start()

@app.route("/detect", methods=["POST"])
def detect():
    # 保留旧的上传识别接口
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "missing image"}), 400
    b64 = data["image"].split(",")[-1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    frame = np.array(img)
    res = recog.recognize(frame)
    res["timestamp"] = datetime.utcnow().isoformat()+"Z"
    return jsonify(res)

@app.route("/status")
def status():
    return jsonify({"ok": True, "faces_loaded": len(recog.known_names)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
