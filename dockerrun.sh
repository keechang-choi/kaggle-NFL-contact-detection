#!/bin/sh

IMAGE_NAME="kaggle/nfl:0.0.1"

docker run --rm -v "`pwd`:/workspace" \
     -w /workspace \
     -it \
     --entrypoint /bin/bash \
     -p 8888:8888 \
     $IMAGE_NAME