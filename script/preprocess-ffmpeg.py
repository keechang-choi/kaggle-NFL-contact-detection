#!/opt/conda/bin/python

import os
import sys
import glob
import numpy as np
import pandas as pd
import random
import math
import gc
import cv2
from tqdm import tqdm
import time
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef

import subprocess

DATA_PATH = "./data"
pd.set_option('display.max_columns', None)


if __name__ == "__main__":
    test_helmets = pd.read_csv(os.path.join(
        DATA_PATH, "test_baseline_helmets.csv"))
    frame_dir = os.path.join(DATA_PATH, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    print(test_helmets.head(10))

    print(test_helmets["video"])
    for video in tqdm(test_helmets.video.unique()):
        print(video)
        if os.path.isfile(os.path.join(frame_dir, video+"_0001.jpg")):
            continue
        if "Endzone2" not in video:
            subprocess.call(["ffmpeg", "-i", os.path.join(DATA_PATH, f"test/{video}"), "-q:v", "2", "-f", "image2", os.path.join(
                frame_dir, f"{video}_%04d.jpg"), "-hide_banner", "-loglevel", "error"])

    train_helmets = pd.read_csv(os.path.join(
        DATA_PATH, "train_baseline_helmets.csv"))
    os.makedirs(frame_dir, exist_ok=True)
    print(train_helmets.head(10))
    print(train_helmets["view"].unique())
    for video in tqdm(train_helmets.video.unique()):
        if os.path.isfile(os.path.join(frame_dir, video+"_0001.jpg")):
            continue
        # Endzone2 가 view 중에 있는데 뭔지 모름 파악 필요.
        if "Endzone2" not in video:
            subprocess.call(["ffmpeg", "-i", os.path.join(DATA_PATH, f"train/{video}"), "-q:v", "2", "-f", "image2", os.path.join(
                frame_dir, f"{video}_%04d.jpg"), "-hide_banner", "-loglevel", "error"])
