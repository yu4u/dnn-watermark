#!/bin/bash

TAG='funabiki/dnn-watermark_nvidia_cuda:8.0-cudnn5-devel-ubuntu16.04'
PROJECT_DIR="$(cd "$(dirname "${0}")/.." || exit; pwd)"
DATASET_DIR="${PROJECT_DIR}/../dataset/dnn-watermark/.keras"

# build
cd "$(dirname "${0}")/.." || exit
DOCKER_BUILDKIT=1 docker build --progress=plain -t ${TAG} docker

# run
docker run -it --rm \
  --shm-size=8g \
  --gpus all \
  -v "${PROJECT_DIR}:/work" \
  -v "${DATASET_DIR}:/root/.keras" \
  -w "/work" \
  "${TAG}"
