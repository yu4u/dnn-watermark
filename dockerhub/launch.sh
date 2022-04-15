#!/bin/bash

TAG='funabiki/dnn-watermark_tensorflow:0.12.1-gpu-py3'
PROJECT_DIR="$(cd "$(dirname "${0}")/.." || exit; pwd)"

# build
cd "$(dirname "${0}")/.." || exit
DOCKER_BUILDKIT=1 docker build --progress=plain -t ${TAG} dockerhub

# run
docker run -it --rm \
  --shm-size=8g \
  --gpus all \
  -v "${PROJECT_DIR}:/work" \
  -v "${PROJECT_DIR}/../dataset/dnn-watermark/.keras:/root/.keras" \
  -w "/work" \
  "${TAG}"
