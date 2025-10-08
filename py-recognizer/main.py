from flask import Flask, request, jsonify
from datetime import datetime
from PIL import Image
import base64, io, numpy as np, cv2

from recognizer import FaceRecognizer

app = Flask(__name__)
recog = FaceRecognizer(faces_dir="faces", embed_dir="embeddings")

def parse_image():
    """从JSON或multipart解析图片，返回RGB numpy"""
    if request.content_type and "application/json" in request.content_type:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return None, "missing image"
        b64 = data["image"].split(",")[-1]
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    elif request.files.get("image"):
        f = request.files["image"]
        img = Image.open(f.stream).convert("RGB")
    else:
        return None, "no image"
    return np.array(img), None

@app.route("/detect", methods=["POST"])
def detect():
    frame, err = parse_image()
    if err:
        return jsonify({"error": err}), 400
    res = recog.recognize(frame)
    res["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return jsonify(res)

@app.route("/status")
def status():
    return jsonify({"ok": True, "faces_loaded": len(recog.known_names)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
