from models.cnn_2_5.version.A.dataset import CNN25DataModule
from models.cnn_2_5.version.C.dataset import CNN25SingleGroundDataModule
from models.cnn_2_5.version.D.dataset import CNN25SingleFrameDataModule


class DataSetFactory():
    def __init__(self):
        pass

    @classmethod
    def get_dataset(self, name, params):
        if name == 'cnn_2_5-A':
            data_module = CNN25DataModule(**params)
        elif name == "cnn_2_5-B":
            # share datamodule with A
            data_module = CNN25DataModule(**params)
        elif name == "cnn_2_5-C":
            data_module = CNN25SingleGroundDataModule(**params)
        elif name == "cnn_2_5-D":
            data_module = CNN25SingleFrameDataModule(**params)
        else:
            raise NotImplementedError()

        return data_module
