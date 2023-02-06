#!/bin/bash

LOAD_PATH=${1:-""}
echo "LOAD_PATH: '$LOAD_PATH'"
pushd ..
wandb login
python ./src/train.py --load_path=$LOAD_PATH &
