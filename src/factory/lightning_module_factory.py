from models.cnn_2_5.version.A.model import CNN25LightningModule
from models.cnn_2_5.version.B.model import CNN25BSLightningModule


class LightningModuleFactory():
    def __init__(self):
        pass

    @classmethod
    def get_lightning_module(cls, name, params):
        lightning_module = LightningModuleFactory.init_lightning_module(
            name, params)

        return lightning_module

    @classmethod
    def init_lightning_module(cls, name, params):
        if name == 'cnn_2_5-A':
            lightning_module = CNN25LightningModule(**params)
        elif name == "cnn_2_5-B":
            lightning_module = CNN25BSLightningModule(**params)
        else:
            raise NotImplementedError()

        return lightning_module

    @classmethod
    def load_lightning_module(cls, name, params, load_path):
        if name == 'cnn_2_5-A':
            # NOTE: backbone 등 params 정보도 load할때 class init에 필요함.
            # https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/core/saving.html#ModelIO.load_from_checkpoint
            lightning_module = CNN25LightningModule.load_from_checkpoint(
                load_path,
                **params)
        elif name == "cnn_2_5-B":
            lightning_module = CNN25BSLightningModule.load_from_checkpoint(
                load_path,
                **params)
        else:
            raise NotImplementedError()

        return lightning_module
