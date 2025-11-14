#!/usr/bin/env python3
import cv2
import os
import time
from datetime import datetime
from evdev import InputDevice, categorize, ecodes
import select

# === 配置部分 ===
CAMERA_INDEX = 0  # 通常 USB 摄像头在 /dev/video0，就是 index=0
BUTTON_EVENT_DEVICE = "/dev/input/event5"  # <<< 把这个换成你实际查到的路径
SAVE_DIR = "~/pro/face/1"

def init_camera(index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 /dev/video{index}")
    # 可按需要调整分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap

def ensure_save_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def capture_and_save(cap, save_dir: str):
    ret, frame = cap.read()
    if not ret:
        print("❌ 拍照失败：无法从摄像头读取图像")
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(save_dir, f"face_{ts}.jpg")
    cv2.imwrite(filename, frame)
    print(f"✅ 已保存图片: {filename}")

def main():
    print("🚀 程序启动：初始化摄像头与按键设备...")
    ensure_save_dir(SAVE_DIR)

    # 初始化摄像头
    cap = init_camera(CAMERA_INDEX)

    # 初始化按键设备
    try:
        dev = InputDevice(BUTTON_EVENT_DEVICE)
    except Exception as e:
        cap.release()
        raise RuntimeError(f"无法打开按键输入设备 {BUTTON_EVENT_DEVICE}，请检查路径是否正确") from e

    print(f"✅ 摄像头已打开：/dev/video{CAMERA_INDEX}")
    print(f"✅ 按键设备已打开：{BUTTON_EVENT_DEVICE} ({dev.name!r})")
    print("📸 按下摄像头上的按键即可拍照，Ctrl+C 退出程序。")

    try:
        while True:
            # 使用 select 等待事件，有事件再处理，避免 CPU 100%
            r, _, _ = select.select([dev.fd], [], [], 1.0)
            if dev.fd in r:
                for event in dev.read():
                    if event.type == ecodes.EV_KEY:
                        key_event = categorize(event)
                        # 只在按下时触发 (event.value == 1 表示按下; 0 松开; 2 长按重复)
                        if event.value == 1:
                            # 可以根据需要只过滤某一个按键，比如 KEY_CAMERA
                            # if key_event.scancode != ecodes.KEY_CAMERA: continue
                            print(f"🔘 检测到按键：{key_event.keycode}，正在拍照...")
                            capture_and_save(cap, SAVE_DIR)
            # 此处可以根据需要添加其它逻辑，比如定时预览等
    except KeyboardInterrupt:
        print("\n🛑 接收到 Ctrl+C，准备退出...")
    finally:
        cap.release()
        print("👌 资源已释放，程序结束。")

if __name__ == "__main__":
    main()
