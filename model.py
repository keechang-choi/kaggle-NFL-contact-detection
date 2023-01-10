import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import timm
from config import CFG


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = timm.create_model(
            CFG['model'], pretrained=False, num_classes=500, in_chans=13)
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
        self.fc_sigmoid = nn.Sequential(
            nn.Linear(64+500*2, 1),
            nn.Sigmoid()
        )

    def forward(self, img, feature):
        b, c, h, w = img.shape
        img = img.reshape(b*2, c//2, h, w)
        img = self.backbone(img).reshape(b, -1)
        feature = self.mlp(feature)
        y = self.fc_sigmoid(torch.cat([img, feature], dim=1))

        return y


class LitCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Model()

    def training_step(self, batch, batch_index):
        img, feature, label = batch
        output = self.model(img, feature).squeeze(-1)
        loss = F.binary_cross_entropy(output, label)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_index):
        img, feature, label = batch
        output = self.model(img, feature).squeeze(-1)
        test_loss = F.binary_cross_entropy(output, label)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_index):
        img, feature, label = batch
        output = self.model(img, feature).squeeze(-1)
        val_loss = F.binary_cross_entropy(output, label)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer