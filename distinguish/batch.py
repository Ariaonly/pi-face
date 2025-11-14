#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量人脸识别脚本：
- 已知人脸：/data/know/<person_name>/*.jpg|*.png
- 未知人脸：/data/unknow/*.jpg|*.png
- 日志输出：/data/logs/records.csv
"""

import os
import csv
import time
from datetime import datetime

import cv2
import numpy as np
import inspireface as isf

# 目录配置（按你 docker 挂载的路径来）
DATA_ROOT = "/data"
KNOWN_DIR = os.path.join(DATA_ROOT, "know")
UNKNOWN_DIR = os.path.join(DATA_ROOT, "unknow")
LOG_DIR = os.path.join(DATA_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "records.csv")

# 支持的图片后缀
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)


def init_session():
    """初始化 InspireFace，全局 launch + 创建 Session"""
    # 触发模型拉取（默认 Pikachu）
    isf.launch()  # 也可以 isf.launch("Pikachu")

    opt = isf.HF_ENABLE_FACE_RECOGNITION
    session = isf.InspireFaceSession(
        param=opt,
        detect_mode=isf.HF_DETECT_MODE_ALWAYS_DETECT,
        max_detect_num=5,
        detect_pixel_level=160,
    )
    session.set_detection_confidence_threshold(0.5)
    return session


def load_known_faces(session):
    """
    从 KNOWN_DIR 加载已知人脸特征
    目录结构：
      /data/know/alice/xxx.jpg
      /data/know/bob/yyy.png
    返回：dict[name] = feature_vector(np.ndarray)
    """
    known_features = {}
    person_count = 0
    img_count = 0

    if not os.path.isdir(KNOWN_DIR):
        print(f"[WARN] 已知人脸目录不存在: {KNOWN_DIR}")
        return known_features

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
            print(f"[INFO] 加载已知人物 {person_name}，使用图片 {len(feats)} 张")
        else:
            print(f"[WARN] 人物 {person_name} 没有可用人脸图像")

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


def recognize_image(session, known_features, image_path, threshold):
    """
    对单张未知图片进行识别，返回 (match_name, similarity, status, msg)
    status: MATCH / NO_MATCH / NO_FACE / ERROR
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, None, "ERROR", "failed to read image"

    faces = session.face_detection(image)
    if not faces:
        return None, None, "NO_FACE", "no face detected"

    face = faces[0]
    feature = session.face_feature_extract(image, face)
    if feature is None:
        return None, None, "ERROR", "feature extraction failed"

    if not known_features:
        return None, None, "ERROR", "no known faces loaded"

    best_name = None
    best_score = -1.0

    for name, known_feat in known_features.items():
        score = isf.feature_comparison(feature, known_feat)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        return best_name, float(best_score), "MATCH", ""
    else:
        return None, float(best_score), "NO_MATCH", ""


def process_unknown_images(session, known_features):
    """遍历 UNKNOWN_DIR，把识别结果写入 CSV 日志"""
    if not os.path.isdir(UNKNOWN_DIR):
        print(f"[WARN] 未知人脸目录不存在: {UNKNOWN_DIR}")
        return

    prepare_log_file()

    # InspireFace 给的推荐阈值
    base_threshold = isf.get_recommended_cosine_threshold()
    print(f"[INFO] 使用推荐阈值: {base_threshold:.4f}")

    total = 0
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for fname in sorted(os.listdir(UNKNOWN_DIR)):
            if not fname.lower().endswith(IMG_EXTS):
                continue

            img_path = os.path.join(UNKNOWN_DIR, fname)
            total += 1
            print(f"[INFO] 处理未知图片 {total}: {img_path}")

            match_name, sim, status, msg = recognize_image(
                session, known_features, img_path, base_threshold
            )

            if status == "MATCH":
                label = match_name
            elif status == "NO_MATCH":
                label = "UNKNOWN"
            elif status == "NO_FACE":
                label = "NO_FACE"
            else:
                label = "ERROR"

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            writer.writerow([
                timestamp,
                img_path,
                label,
                f"{sim:.6f}" if sim is not None else "",
                f"{base_threshold:.6f}",
                status,
                msg,
            ])
            f.flush()

    print(f"[INFO] 处理完成，未知图片总数: {total}")
    print(f"[INFO] 日志已写入: {LOG_FILE}")


def main():
    start = time.time()
    ensure_dirs()
    print("[INFO] 初始化 InspireFace...")
    session = init_session()
    print("[INFO] 加载已知人脸特征...")
    known_features = load_known_faces(session)
    print("[INFO] 开始处理未知人脸...")
    process_unknown_images(session, known_features)
    cost = time.time() - start
    print(f"[INFO] 全部完成，用时 {cost:.2f} 秒")


if __name__ == "__main__":
    main()
