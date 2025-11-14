#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单文件人脸识别运行脚本：

- 已知人脸：/data/know
    支持两种结构：
      1) /data/know/<person_name>/*.jpg|*.png
      2) /data/know/*.jpg|*.png （用文件名前缀当人名）

- 按钮：/dev/input/event5  （EV_KEY 任意按下触发）

- 摄像头：/dev/video0  （OpenCV: index=0）

- 未知抓拍：/data/unknow/UNKNOWN_YYYYmmdd_HHMMSS_xxx.jpg

- 日志：/data/logs/records.csv
    字段：
      timestamp, image_path, match_name, similarity,
      threshold, status, message

防堵塞：
- 主线程：监听按键 + 拍照 + 把任务放到队列
- 工作线程：从队列取帧 → 识别 → 写日志
"""

import os
import csv
import time
import threading
import queue
from datetime import datetime

import cv2
import numpy as np
import inspireface as isf
from evdev import InputDevice, categorize, ecodes


# ===== 路径配置（全部使用绝对路径） =====
DATA_ROOT = "/data"
KNOWN_DIR = os.path.join(DATA_ROOT, "know")
UNKNOWN_DIR = os.path.join(DATA_ROOT, "unknow")
LOG_DIR = os.path.join(DATA_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "records.csv")

EVENT_DEVICE_PATH = "/dev/input/event5"  # 按钮事件设备
CAMERA_INDEX = 0                         # /dev/video0

# 支持的图片后缀
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# 任务队列最大长度（防止无限堆积）
TASK_QUEUE_MAXSIZE = 10


def ensure_dirs():
    """确保必要目录存在"""
    os.makedirs(KNOWN_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def init_session():
    """初始化 InspireFace：拉取模型 + 创建 Session"""
    print("[INFO] 初始化 InspireFace (launch Pikachu)...")
    isf.launch()  # 默认就是 Pikachu

    opt = isf.HF_ENABLE_FACE_RECOGNITION
    session = isf.InspireFaceSession(
        param=opt,
        detect_mode=isf.HF_DETECT_MODE_ALWAYS_DETECT,
        max_detect_num=5,
        detect_pixel_level=160,
    )
    session.set_detection_confidence_threshold(0.5)
    print("[INFO] InspireFace Session 初始化完成")
    return session


def load_known_faces(session):
    """
    从 KNOWN_DIR 加载已知人脸特征
    支持两种结构：
      1) /data/know/<person_name>/*.jpg|*.png
      2) /data/know/*.jpg|*.png  （用文件名前缀当人名）
    返回：dict[name] = feature_vector(np.ndarray)
    """
    known_features = {}
    person_count = 0
    img_count = 0

    if not os.path.isdir(KNOWN_DIR):
        print(f"[WARN] 已知人脸目录不存在: {KNOWN_DIR}")
        return known_features

    # ---------- 模式 1：子目录，每个目录一个人 ----------
    for person_name in sorted(os.listdir(KNOWN_DIR)):
        person_dir = os.path.join(KNOWN_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        feats = []
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            img_path = os.path.join(person_dir, fname)
            image = cv2.imread(img_path)
            if image is None:
                print(f"[WARN] 无法读取图像: {img_path}")
                continue

            faces = session.face_detection(image)
            if not faces:
                print(f"[WARN] 未检测到人脸: {img_path}")
                continue

            face = faces[0]
            feature = session.face_feature_extract(image, face)
            if feature is None:
                print(f"[WARN] 提取特征失败: {img_path}")
                continue

            feats.append(feature)
            img_count += 1

        if feats:
            feature_avg = np.mean(np.stack(feats, axis=0), axis=0)
            known_features[person_name] = feature_avg
            person_count += 1
            print(f"[INFO] [子目录] 加载已知人物 {person_name}，使用图片 {len(feats)} 张")
        else:
            print(f"[WARN] [子目录] 人物 {person_name} 没有可用人脸图像")

    # ---------- 模式 2：根目录下直接放图片 ----------
    root_feats = {}  # name -> list[feat]

    for fname in sorted(os.listdir(KNOWN_DIR)):
        path = os.path.join(KNOWN_DIR, fname)
        if not os.path.isfile(path):
            continue
        if not fname.lower().endswith(IMG_EXTS):
            continue

        # 取文件名（去扩展名），遇到 "_" 或 "-" 前面的作为人名
        base = os.path.splitext(fname)[0]
        for sep in ["_", "-"]:
            if sep in base:
                base = base.split(sep)[0]
                break
        person_name = base

        image = cv2.imread(path)
        if image is None:
            print(f"[WARN] 无法读取图像: {path}")
            continue

        faces = session.face_detection(image)
        if not faces:
            print(f"[WARN] 未检测到人脸: {path}")
            continue

        face = faces[0]
        feature = session.face_feature_extract(image, face)
        if feature is None:
            print(f"[WARN] 提取特征失败: {path}")
            continue

        root_feats.setdefault(person_name, []).append(feature)
        img_count += 1

    for person_name, feats in root_feats.items():
        feature_avg = np.mean(np.stack(feats, axis=0), axis=0)
        # 如果子目录模式里已经有同名人物，这里就“补充平均”
        if person_name in known_features:
            feature_avg = np.mean(
                np.stack([known_features[person_name], feature_avg], axis=0),
                axis=0
            )
        known_features[person_name] = feature_avg
        person_count += 1
        print(f"[INFO] [根目录] 加载已知人物 {person_name}，使用图片 {len(feats)} 张")

    print(f"[INFO] 已加载已知人物数量: {person_count}，总有效图片: {img_count}")
    return known_features


def prepare_log_file():
    """如果日志不存在，则写入表头"""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",      # 时间戳
                "image_path",     # 未知图片路径
                "match_name",     # 匹配到的人名 (或 UNKNOWN/NO_FACE/ERROR)
                "similarity",     # 相似度（余弦）
                "threshold",      # 使用的阈值
                "status",         # MATCH / NO_MATCH / NO_FACE / ERROR
                "message",        # 额外信息
            ])


def recognize_and_log(session, known_features, frame, image_path, timestamp_str, threshold):
    """
    在工作线程中执行：识别 + 写日志
    """
    status = "ERROR"
    label = "ERROR"
    msg = ""
    sim = None

    try:
        if frame is None:
            status = "ERROR"
            msg = "frame is None"
        else:
            faces = session.face_detection(frame)
            if not faces:
                status = "NO_FACE"
                label = "NO_FACE"
                msg = "no face detected"
            else:
                face = faces[0]
                feature = session.face_feature_extract(frame, face)
                if feature is None:
                    status = "ERROR"
                    label = "ERROR"
                    msg = "feature extraction failed"
                elif not known_features:
                    status = "ERROR"
                    label = "ERROR"
                    msg = "no known faces loaded"
                else:
                    best_name = None
                    best_score = -1.0
                    for name, known_feat in known_features.items():
                        score = isf.feature_comparison(feature, known_feat)
                        if score > best_score:
                            best_score = score
                            best_name = name

                    sim = float(best_score)

                    if best_score >= threshold:
                        status = "MATCH"
                        label = best_name
                    else:
                        status = "NO_MATCH"
                        label = "UNKNOWN"

    except Exception as e:
        status = "ERROR"
        label = "ERROR"
        msg = f"exception: {e}"

    # 写日志（每次 append 一行，避免多线程共享 file handle）
    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp_str,
                image_path,
                label,
                f"{sim:.6f}" if sim is not None else "",
                f"{threshold:.6f}",
                status,
                msg,
            ])
    except Exception as e:
        print(f"[ERROR] 写日志失败: {e}")

    print(f"[INFO] 识别完成: {image_path} -> {label} "
          f"(status={status}, sim={sim}, threshold={threshold})")


def worker_loop(session, known_features, task_queue, threshold):
    """
    工作线程主循环：从队列取任务，调用 recognize_and_log
    任务格式：(frame, image_path, timestamp_str)
    """
    print("[INFO] 工作线程已启动，等待识别任务...")
    while True:
        item = task_queue.get()
        if item is None:
            # 收到结束信号
            break
        frame, image_path, timestamp_str = item
        recognize_and_log(session, known_features, frame, image_path, timestamp_str, threshold)
        task_queue.task_done()


def open_camera():
    """打开摄像头 /dev/video0"""
    print("[INFO] 打开摄像头 /dev/video0 ...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头 /dev/video0")
    print("[INFO] 摄像头已打开")
    return cap


def open_input_device():
    """打开按键输入设备 /dev/input/event5"""
    print(f"[INFO] 打开输入设备 {EVENT_DEVICE_PATH} ...")
    dev = InputDevice(EVENT_DEVICE_PATH)
    print(f"[INFO] 输入设备已打开: {dev}")
    return dev


def main():
    ensure_dirs()
    prepare_log_file()

    # 初始化 InspireFace
    session = init_session()
    # 加载已知人脸
    known_features = load_known_faces(session)

    # 推荐阈值（可以后期根据现场数据微调）
    threshold = isf.get_recommended_cosine_threshold()
    print(f"[INFO] 使用推荐阈值: {threshold:.4f}")

    # 打开摄像头 & 输入设备
    cap = open_camera()
    dev = open_input_device()

    # 创建任务队列 & 工作线程
    task_queue = queue.Queue(maxsize=TASK_QUEUE_MAXSIZE)
    worker = threading.Thread(
        target=worker_loop,
        args=(session, known_features, task_queue, threshold),
        daemon=True,
    )
    worker.start()

    print("[INFO] 进入主循环：等待按键事件触发拍照...")

    try:
        for event in dev.read_loop():
            if event.type != ecodes.EV_KEY:
                continue

            key_event = categorize(event)
            # 只在“按下”时触发（keystate: 1=down, 2=hold, 0=up）
            if key_event.keystate not in (key_event.key_down, key_event.key_hold):
                continue

            # 这里你可以打印一下是哪个键：
            # print(f"[DEBUG] key: {key_event.keycode}, keystate: {key_event.keystate}")

            # 拍照
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] 摄像头抓拍失败")
                # 写一条错误日志
                now = datetime.now()
                timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
                image_path = ""  # 没有保存成功
                with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp_str,
                        image_path,
                        "ERROR",
                        "",
                        f"{threshold:.6f}",
                        "ERROR",
                        "camera capture failed",
                    ])
                continue

            # 保存图片到 UNKNOWN_DIR
            now = datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
            fname_ts = now.strftime("%Y%m%d_%H%M%S")
            filename = f"UNKNOWN_{fname_ts}.jpg"
            image_path = os.path.join(UNKNOWN_DIR, filename)

            try:
                cv2.imwrite(image_path, frame)
            except Exception as e:
                print(f"[ERROR] 保存图片失败: {e}")
                # 记录日志
                with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp_str,
                        image_path,
                        "ERROR",
                        "",
                        f"{threshold:.6f}",
                        "ERROR",
                        f"save image failed: {e}",
                    ])
                continue

            print(f"[INFO] 按键触发拍照，已保存到: {image_path}")

            # 把任务提交给队列（工作线程异步识别）
            try:
                task_queue.put_nowait((frame.copy(), image_path, timestamp_str))
            except queue.Full:
                print("[WARN] 任务队列已满，本次识别任务被丢弃")
                # 写一条 ERROR 日志说明队列满
                with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp_str,
                        image_path,
                        "ERROR",
                        "",
                        f"{threshold:.6f}",
                        "ERROR",
                        "task queue full, dropped",
                    ])

    except KeyboardInterrupt:
        print("\n[INFO] 收到中断信号，准备退出...")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        # 给工作线程发一个结束信号（如果你想优雅退出，可以取消注释下面两行）
        # task_queue.put(None)
        # worker.join(timeout=1.0)
        print("[INFO] 已退出主循环")


if __name__ == "__main__":
    main()
