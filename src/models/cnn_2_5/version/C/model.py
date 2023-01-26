import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import timm
from config import CFG
import torchmetrics

from utils.mcc import MCC_Loss


class CNN25SingleGroundModel(nn.Module):
    def __init__(self, backbone):
        super(CNN25SingleGroundModel, self).__init__()
        in_chans = CFG["window"] // 4 * 2 + 1
        resnet = timm.create_model(
            backbone,
            pretrained=True)
        self.backbone = nn.Sequential(**list(resnet.children())[:-1])

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
        # NOTE: linear layer 구성 바꿔서 실험. layer 추가 등.
        # for binary cross entropy, added sigmoid activation
        self.fc_sigmoid = nn.Sequential(
            nn.Linear(64+2048, 1),
            nn.Sigmoid()
        )

    def forward(self, img, feature):
        # c = rgb 3채널
        b, v, c, h, w = img.shape
        # batch와 view를 바꿔서 넣어줌
        img = img.transpose(0, 1)

        view_pool = []
        for each_view in img:
            each_view = self.backbone(each_view)
            # batch_size x 2048
            # each_view = each_view.view(each_view.size(0), -1)
            view_pool.append(each_view)

        pooled_view = view_pool[0]
        # view max pooling before fc
        for i in range(1, len(view_pool)):
            pooled_view = torch.max(pooled_view, view_pool[i])

        feature = self.mlp(feature)
        y = self.fc_sigmoid(torch.cat([pooled_view, feature], dim=1))

        return y


class CNN25SingleGroundLightningModule(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.model = CNN25SingleGroundModel(backbone)

        self.valid_acc = torchmetrics.Accuracy(
            task='binary', threshold=CFG["threshold"])
        self.test_acc = torchmetrics.Accuracy(
            task='binary', threshold=CFG["threshold"])
        self.mcc_loss = MCC_Loss()
        self.last_test_output = None

    def training_step(self, batch, batch_index):
        img, feature, label = batch
        output = self.model(img, feature).squeeze(-1)
        loss = F.binary_cross_entropy(output, label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_index):
        img, feature, label = batch
        output = self.model(img, feature).squeeze(-1)
        val_loss = F.binary_cross_entropy(output, label)
        self.log("val_loss", val_loss)

        self.valid_acc(output, label)
        self.log('valid_acc_step', self.valid_acc)

    def test_step(self, batch, batch_index):
        img, feature, label = batch
        output = self.model(img, feature).squeeze(-1)
        test_loss = F.binary_cross_entropy(output, label)
        self.log("test_loss", test_loss)

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
        return output
