import os
import random
import torch
import numpy as np
import pytorch_lightning as pl

from config import CFG
from dataset import MyDataModule
from model import LitCNN


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    seed_everything(CFG['seed'])
    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"

    device = torch.device(device_str)
    data_module = MyDataModule(data_dir="data")
    model = LitCNN()

    trainer = pl.Trainer(max_epochs=CFG["epochs"],
                         accelerator="gpu",
                         devices=1 if device_str != "cpu" else None)
    trainer.fit(model=model, datamodule=data_module)
