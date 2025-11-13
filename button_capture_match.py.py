cat >/app/button_capture_match.py << 'PY'
import os
import time
import csv
import argparse

import cv2
import face_recognition
from evdev import InputDevice, categorize, ecodes, list_devices

KNOWN_DIR = "/data/1"
UNKNOWN_DIR = "/data/2"
LOG_DIR = "/data/logs"
LOG_FILE = os.path.join(LOG_DIR, "records.csv")


def parse_args():
    p = argparse.ArgumentParser(description="按键拍照 + 与已知人脸对比 + 写入日志")
    p.add_argument("--input", default=None,
                   help="可选：直接指定按键设备，如 /dev/input/event3；如果不指定则自动尝试识别")
    p.add_argument("--key-code", type=int, default=None,
                   help="可选：只在指定按键 code 时触发，例如 212=KEY_CAMERA")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="匹配阈值，值越小越严格，默认 0.5")
    return p.parse_args()


def auto_detect_input_device():
    """
    自动寻找可能属于摄像头的按键设备：
    1. 优先按设备名里包含 Camera / camera / HD / Lens 等关键字
    2. 其次：找包含 KEY_CAMERA 的设备
    找不到则返回 None
    """
    print("[INFO] 尝试自动识别按键设备...")
    candidates = []

    # 1) 根据名字筛选
    for path in list_devices():
        dev = InputDevice(path)
        name = dev.name or ""
        if any(k in name for k in ["Camera", "camera", "HD", "Lens", "G-Lens", "Defender"]):
            print(f"[INFO] 按名字匹配到候选设备: {path} ({name})")
            candidates.append(dev)

    if candidates:
        chosen = candidates[0]
        print(f"[INFO] 选用第一个候选设备: {chosen.path} ({chosen.name})")
        return chosen.path

    # 2) 找包含 KEY_CAMERA 的设备
    for path in list_devices():
        dev = InputDevice(path)
        name = dev.name or ""
        caps = dev.capabilities(verbose=True)
        for etype, items in caps.items():
            if etype[0] != "EV_KEY":
                continue
            for code in items:
                key_name = code[0]
                if key_name == "KEY_CAMERA":
                    print(f"[INFO] 按 KEY_CAMERA 匹配到设备: {path} ({name})")
                    return path

    print("[WARN] 无法自动识别按键设备，请考虑手动使用 --input 参数指定 /dev/input/eventX")
    return None


def load_known_faces():
    """
    从 /data/1 下加载已知人脸：
    规则：每个图片文件名（不含扩展名）就是人物名称。
    例如 /data/1/zhangsan.jpg -> 'zhangsan'
    """
    known_encodings = []
    known_labels = []

    if not os.path.isdir(KNOWN_DIR):
        print(f"[WARN] 已知人脸目录不存在: {KNOWN_DIR}")
        return known_encodings, known_labels

    print(f"[INFO] 开始从 {KNOWN_DIR} 加载已知人脸（文件名=人名）...")
    for filename in sorted(os.listdir(KNOWN_DIR)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        label = os.path.splitext(filename)[0]  # 文件名部分
        img_path = os.path.join(KNOWN_DIR, filename)

        try:
            image = face_recognition.load_image_file(img_path)
            locations = face_recognition.face_locations(image, model="hog")
            if not locations:
                print(f"[WARN] {img_path} 中未检测到人脸，跳过")
                continue
            encs = face_recognition.face_encodings(image, known_face_locations=locations)
            if not encs:
                print(f"[WARN] {img_path} 无法提取特征，跳过")
                continue

            known_encodings.append(encs[0])
            known_labels.append(label)
            print(f"[OK] 已加载: {img_path} -> {label}")
        except Exception as e:
            print(f"[ERROR] 处理 {img_path} 时异常: {e}")

    print(f"[INFO] 已加载已知人脸数: {len(known_encodings)}")
    return known_encodings, known_labels


def ensure_log_header():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",   # 时间戳
                "image_path",  # /data/2 下的图片路径
                "face_index",  # 第几张人脸（从 0 开始）
                "matched",     # 是否匹配(1/0)
                "name",        # 匹配到的名字或 unknown / NO_FACE 等
                "distance",    # 距离（越小越像）
                "num_known"    # 已知人脸总数
            ])


def log_result(rows):
    ensure_log_header()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def match_faces_for_image(image_path, known_encodings, known_labels, threshold):
    rows = []
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        image = face_recognition.load_image_file(image_path)
    except Exception as e:
        print(f"[ERROR] 读取图片失败 {image_path}: {e}")
        rows.append([
            timestamp, image_path, -1, 0, "READ_ERROR", "", len(known_encodings)
        ])
        return rows

    locations = face_recognition.face_locations(image, model="hog")
    if not locations:
        print(f"[INFO] {image_path} 中没有检测到人脸")
        rows.append([
            timestamp, image_path, -1, 0, "NO_FACE", "", len(known_encodings)
        ])
        return rows

    encs = face_recognition.face_encodings(image, known_face_locations=locations)
    if not encs:
        print(f"[INFO] {image_path} 无法提取人脸特征")
        rows.append([
            timestamp, image_path, -1, 0, "NO_ENCODING", "", len(known_encodings)
        ])
        return rows

    if not known_encodings:
        print(f"[WARN] 当前没有已知人脸，全部记为 unknown")
        for idx, _ in enumerate(encs):
            rows.append([
                timestamp, image_path, idx, 0, "unknown", "", 0
            ])
        return rows

    import numpy as np
    for idx, enc in enumerate(encs):
        distances = face_recognition.face_distance(known_encodings, enc)
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        is_match = best_distance < threshold
        name = known_labels[best_index] if is_match else "unknown"

        print(f"[MATCH] face #{idx}, best={name}, distance={best_distance:.4f}, match={is_match}")

        rows.append([
            timestamp,
            image_path,
            idx,
            int(is_match),
            name,
            f"{best_distance:.6f}",
            len(known_encodings),
        ])
    return rows


def main():
    args = parse_args()

    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    os.makedirs(KNOWN_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 预加载已知人脸
    known_encodings, known_labels = load_known_faces()

    # 摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Cannot open /dev/video0")

    # 确定按键设备
    if args.input:
        input_path = args.input
        print(f"[INFO] 使用指定按键设备: {input_path}")
    else:
        input_path = auto_detect_input_device()
        if not input_path:
            raise SystemExit("无法自动识别按键设备，请手动用 --input 指定 /dev/input/eventX")

    dev = InputDevice(input_path)
    print(f"[INFO] 监听按键设备: {dev.path} ({dev.name})")
    print(f"[INFO] 已知人脸目录: {KNOWN_DIR}")
    print(f"[INFO] 未知人脸目录: {UNKNOWN_DIR}")
    print(f"[INFO] 日志文件: {LOG_FILE}")
    print("[INFO] 容器已启动，按下摄像头物理按键即可拍照+识别")

    shot_id = 0

    for event in dev.read_loop():
        if event.type != ecodes.EV_KEY:
            continue

        keyevent = categorize(event)
        if keyevent.keystate != keyevent.key_down:
            continue

        code = keyevent.scancode
        print(f"[KEY] code={code}, key={keyevent.keycode}")

        if args.key_code is not None and code != args.key_code:
            continue

        # 拍照
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 无法从摄像头读取图像，拍照失败")
            continue

        ts_file = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts_file}_{shot_id:04d}.jpg"
        filepath = os.path.join(UNKNOWN_DIR, filename)
        cv2.imwrite(filepath, frame)
        print(f"[CAPTURE] 保存图片 -> {filepath}")

        # 对这张图做识别并记录日志
        rows = match_faces_for_image(filepath, known_encodings, known_labels, args.threshold)
        log_result(rows)
        print(f"[LOG] 已写入 {len(rows)} 条记录到 {LOG_FILE}")

        shot_id += 1


if __name__ == "__main__":
    main()
PY
