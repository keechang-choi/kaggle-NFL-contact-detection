CFG = {
    'model_name': 'cnn_2_5',
    'model_version': 'A',
    'dataset_params': {
        'data_dir': '../data',
    },
    'model_params': {
        'backbone':'resnet50'
    },
    'seed': 42,
    'img_size': 256,
    'epochs': 10,
    'train_bs': 100,
    'valid_bs': 64,
    'lr': 1e-3,
    'weight_decay': 1e-6,
    'num_workers': 8,
    "batch_size": 16
}
