#!/bin/bash

python train.py & tensorboard --logdir ./lightning_logs --host 0.0.0.0 --port 6006 && fg
