CFG = {
    'exp_name': '',
    'model_name': 'cnn_2_5',
    'model_version': 'D',
    'dataset_params': {
        'data_dir': './data',
        "preprocess_result_dir": "./data/processed",
        # all, ground-only, players
        "data_filter": "ground-only"
    },
    'model_params': {
        'backbone': 'resnet50',
        "pos_weight": 9.195  # 1.0
    },
    "num_train_video": 240,
    'seed': 42,
    'img_size': 256,
    'epochs': 10,
    'lr': 1e-3,
    'weight_decay': 1e-6,
    'num_workers': 8,
    "batch_size": 24,
    "logger_dir": "../lightning_logs",
    "threshold": 0.29,
    "is_prediction": False,
    "reproduce_processed_data": False,
    "window": 24
}
