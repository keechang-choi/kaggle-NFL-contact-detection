#!/bin/sh

IMAGE_NAME="kaggle/nfl:0.0.1"
cd ..
docker build --force-rm --network host -t $IMAGE_NAME . -f Dockerfile --build-arg UID=$(id -u)
cd -
