import os
import random
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from config import CFG
from factory.dataset_factory import DataSetFactory
from factory.lightning_module_factory import LightningModuleFactory


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
    parser = argparse.ArgumentParser(description="training args")
    parser.add_argument("--load_path", type=str, default="")
    args = parser.parse_args()

    seed_everything(CFG['seed'])
    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
    device = torch.device(device_str)

    logger_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), CFG["logger_dir"]))
    os.makedirs(logger_path, exist_ok=True)
    logger = TensorBoardLogger(logger_path, name=CFG["model_name"])

    model_name = f"{CFG['model_name']}-{CFG['model_version']}"
    dataset_params = CFG['dataset_params']
    model_params = CFG['model_params']
    data_module = DataSetFactory.get_dataset(name=model_name,
                                             params=dataset_params)
    lightning_module = LightningModuleFactory.get_lightning_module(name=model_name,
                                                                   load_path=args.load_path,
                                                                   params=model_params)

    # NOTE: cuda, mps, cpu for accelerator.
    # https://pytorch-lightning.readthedocs.io/en/stable/accelerators/mps_basic.html
    trainer = pl.Trainer(max_epochs=CFG["epochs"],
                         accelerator=device_str,
                         devices=1 if device_str != "cpu" else None,
                         logger=logger,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3)])
    trainer.fit(model=lightning_module, datamodule=data_module)