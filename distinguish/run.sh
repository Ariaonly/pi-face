
docker run --rm -it \
  --name=face \
  --device=/dev/video0 \
  --device=/dev/input/event5 \
  --mount type=bind,source="$HOME/pro/face",target=/data \
  --network host \
  face:pi5 \
  /bin/bash
