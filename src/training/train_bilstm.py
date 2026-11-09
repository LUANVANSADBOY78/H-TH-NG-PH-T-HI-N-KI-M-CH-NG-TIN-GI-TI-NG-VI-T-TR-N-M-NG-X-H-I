"""
Kịch bản Huấn luyện Mô hình BiLSTM + Attention Mechanism (PyTorch).
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import BILSTM_DIR, PROCESSED_DATA_DIR
from src.deep_learning_models import BiLSTMClassifier
from src.training.hyperparameter import BILSTM_CONFIG

def simple_vocab_encode(texts, max_len=128, vocab_size=10000):
    word_to_id = {"<pad>": 0, "<unk>": 1}
    curr_id = 2
    for t in texts:
        for w in str(t).split():
            if w not in word_to_id and curr_id < vocab_size:
                word_to_id[w] = curr_id
                curr_id += 1

    encoded = []
    for t in texts:
        tokens = [word_to_id.get(w, 1) for w in str(t).split()][:max_len]
        padded = tokens + [0] * (max_len - len(tokens))
        encoded.append(padded)
    return np.array(encoded), word_to_id

def train_bilstm_pipeline(config=BILSTM_CONFIG):
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    if not os.path.exists(train_path):
        print("Chưa có train.csv, vui lòng chạy split_data.py trước!")
        return

    train_df = pd.read_csv(train_path)
    text_col = "text_seg" if "text_seg" in train_df.columns else "text"

    X_train, vocab = simple_vocab_encode(train_df[text_col])
    y_train = train_df["label"].values

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model = BiLSTMClassifier(vocab_size=len(vocab) + 1, embed_dim=config["embed_dim"], hidden_dim=config["hidden_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    model.train()
    print(">>> Huấn luyện BiLSTM...")
    for epoch in range(1, 4):  # Chạy thử nghiệm 3 epoch
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: Loss = {total_loss/len(loader):.4f}")

    save_path = os.path.join(BILSTM_DIR, "bilstm_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"-> Đã lưu mô hình BiLSTM tại: {save_path}")

if __name__ == "__main__":
    train_bilstm_pipeline()

