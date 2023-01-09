#!/bin/sh

IMAGE_NAME="kaggle/nfl:0.0.1"

docker run --rm -v "`pwd`:/workspace" \
     -v ~/.ssh:/home/docker_user/.ssh:ro \
     -w /workspace \
     -it \
     --entrypoint /bin/bash \
     -p 8888:8888 \
     --gpus '"device=0"' \
     $IMAGE_NAME
