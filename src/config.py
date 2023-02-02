CFG = {
    'model_name': 'cnn_2_5',
    'model_version': 'C',
    'dataset_params': {
        'data_dir': './data',
        "preprocess_result_dir": "./data/processed",
        # all, ground-only, players
        "data_filter": "players"
    },
    'model_params': {
        'backbone': 'resnet50',
        "pos_weight": 9.195  # 1.0
    },
    "num_train_video": 240,
    'seed': 42,
    'img_size': 256,
    'epochs': 15,
    'lr': 1e-3,
    'weight_decay': 1e-6,
    'num_workers': 16,
    "batch_size": 32,
    "logger_dir": "../lightning_logs",
    "threshold": 0.29,
    "is_prediction": False,
    "reproduce_processed_data": False,
    "window": 24
}
