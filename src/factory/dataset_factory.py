from models.cnn_2_5.version.A.dataset import CNN25DataModule


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
        else:
            raise NotImplementedError()

        return data_module
