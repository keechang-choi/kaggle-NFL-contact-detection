import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import timm
from config import CFG
import torchmetrics

from utils.loss import MCC_Loss


class CNN25Model(nn.Module):
    def __init__(self, backbone):
        super(CNN25Model, self).__init__()
        in_chans = CFG["window"] // 4 * 2 + 1
        self.backbone = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=500,
            in_chans=in_chans)

        self.mlp = nn.Sequential(
            nn.Linear(18, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(64, 64),
            # nn.LayerNorm(64),
            # nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64+500*2, 1),
        )

    def forward(self, img, feature):
        b, c, h, w = img.shape
        img = img.reshape(b*2, c//2, h, w)
        img = self.backbone(img).reshape(b, -1)
        feature = self.mlp(feature)
        y = self.fc(torch.cat([img, feature], dim=1))

        return y


class CNN25LightningModule(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.model = CNN25Model(backbone)

        self.valid_acc = torchmetrics.Accuracy(
            task='binary', threshold=CFG["threshold"])
        self.test_acc = torchmetrics.Accuracy(
            task='binary', threshold=CFG["threshold"])
        self.mcc_loss = MCC_Loss()
        self.last_test_output = None
        self.loss_function = nn.BCEWithLogitsLoss()
        self.sigmoid_function = nn.Sigmoid()

    def training_step(self, batch, batch_index):
        img, feature, label = batch
        output = self.model(img, feature).squeeze(-1)
        loss = self.loss_function(output, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_index):
        img, feature, label = batch
        output = self.model(img, feature).squeeze(-1)
        val_loss = self.loss_function(output, label)
        self.log("val_loss", val_loss)

        output = self.sigmoid_function(output)
        self.valid_acc(output, label)
        self.log('valid_acc_step', self.valid_acc)

    def test_step(self, batch, batch_index):
        img, feature, label = batch
        output = self.model(img, feature).squeeze(-1)
        test_loss = self.loss_function(output, label)
        self.log("test_loss", test_loss)

        output = self.sigmoid_function(output)
        self.test_acc(output, label)
        self.log('test_acc_step', self.test_acc)
        return torch.stack((output, label), dim=1)

    def test_epoch_end(self, outputs) -> None:
        self.last_test_output = outputs
        outputs_cat = torch.cat(outputs)
        y, labels = outputs_cat[:, 0], outputs_cat[:, 1]
        preds = (y > CFG["threshold"]).float()
        print(
            f"====== [Test contact counts] ======\n"
            f"-- preds  : {torch.sum(preds)}/{len(preds)}\n"
            f"-- labels : {torch.sum(labels)}/{len(labels)}\n")
        mcc_loss = self.mcc_loss(preds, labels)
        mcc = 1.0 - mcc_loss
        self.log("test_mcc", mcc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img, feature, label = batch
        output = self.model(img, feature).squeeze(-1)
        output = self.sigmoid_function(output)
        return output
