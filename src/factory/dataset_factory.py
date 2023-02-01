from models.cnn_2_5.version.A.dataset import CNN25DataModule
from models.cnn_2_5.version.C.dataset import CNN25SingleGroundDataModule


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
        else:
            raise NotImplementedError()

        return data_module
