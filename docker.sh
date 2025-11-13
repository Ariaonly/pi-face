docker run -it --name face \
  --device=/dev/video0 \
  --device=/dev/input/event5 \
  --mount type=bind,source="$HOME/pro/face",target=/data \
  --network host \
  face_recog:pi5 \
  bash

# 如何查看摄像头上的按键的事件设备号
