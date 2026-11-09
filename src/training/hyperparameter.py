"""
Cấu hình và Siêu tham số Huấn luyện (Hyperparameter Configurations).
"""

PHOBERT_CONFIG = {
    "model_name": "vinai/phobert-base-v2",
    "learning_rate": 2e-5,
    "batch_size": 8,
    "epochs": 3,
    "max_length": 192,
    "weight_decay": 0.01,
    "warmup_steps": 50,
    "early_stopping_patience": 2
}

BILSTM_CONFIG = {
    "embed_dim": 128,
    "hidden_dim": 128,
    "num_layers": 2,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 10,
    "dropout": 0.3
}

LSTM_CONFIG = {
    "embed_dim": 128,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 10,
    "dropout": 0.2
}

