
docker run --rm -it \
  --name=face1 \
  --platform linux/arm64 \
  --device=/dev/video0 \
  --mount type=bind,source="$HOME/project/data/face",target=/data \
  --network host \
  face_pikachu_demo1:pi5 \
  /bin/bash
