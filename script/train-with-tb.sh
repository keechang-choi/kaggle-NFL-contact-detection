#!/bin/bash

LOAD_PATH=${1:-""}
echo "LOAD_PATH: '$LOAD_PATH'"

python ../src/train.py --load_path=$LOAD_PATH &
tensorboard --logdir ../lightning_logs --host 0.0.0.0 --port 6006 && fg
