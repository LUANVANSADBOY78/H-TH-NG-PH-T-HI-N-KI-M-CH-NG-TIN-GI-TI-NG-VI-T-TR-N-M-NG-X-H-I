"""
Module Phân chia Dữ liệu Phân tầng (Stratified Train / Val / Test Splitting).
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.dataset.clean_data import clean_text, segment_text
from src.dataset.labeling import standardize_label

def prepare_and_split(
    raw_path: str = "data/raw/data.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(raw_path)
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    
    # Chuẩn hóa nhãn
    df["label"] = df["label"].apply(standardize_label)

    # Làm sạch & tách từ
    df["text_clean"] = df["text"].apply(clean_text)
    df["text_seg"] = df["text_clean"].apply(segment_text)

    # Chia stratified
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=random_seed)
    adj_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=adj_val_size, stratify=train_val["label"], random_state=random_seed)

    train.to_csv(os.path.join(output_dir, "train.csv"), index=False, encoding="utf-8")
    val.to_csv(os.path.join(output_dir, "val.csv"), index=False, encoding="utf-8")
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False, encoding="utf-8")

    return {
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test)
    }

if __name__ == "__main__":
    res = prepare_and_split()
    print("Split completed:", res)

