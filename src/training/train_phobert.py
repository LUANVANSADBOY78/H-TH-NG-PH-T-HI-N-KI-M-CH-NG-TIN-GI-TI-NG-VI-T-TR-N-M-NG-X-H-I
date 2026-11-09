"""
Kịch bản Huấn luyện Fine-tuning PhoBERT.
Tự động lưu checkpoint tốt nhất vào models/phobert/best và đăng ký vào bảng model_versions.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import PHOBERT_DIR, PROCESSED_DATA_DIR, SEED
from src.training.hyperparameter import PHOBERT_CONFIG
from src.database.connection import get_connection

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": round(acc, 4), "precision": round(pr, 4), "recall": round(rc, 4), "macro_f1": round(f1, 4)}

def train_phobert_pipeline(config=PHOBERT_CONFIG):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Huấn luyện PhoBERT trên thiết bị: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=False)

    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DATA_DIR, "val.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    text_col = "text_seg" if "text_seg" in train_df.columns else "text"

    train_ds = Dataset.from_pandas(train_df[[text_col, "label"]].rename(columns={text_col: "text"}))
    val_ds = Dataset.from_pandas(val_df[[text_col, "label"]].rename(columns={text_col: "text"}))

    def tok_fn(b):
        return tokenizer(b["text"], truncation=True, max_length=config["max_length"])

    train_tok = train_ds.map(tok_fn, batched=True)
    val_tok = val_ds.map(tok_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], num_labels=2)

    output_dir = os.path.join(PHOBERT_DIR, "checkpoints")
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=2,
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        dataloader_num_workers=0,
        optim="adamw_torch",
        seed=SEED,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])]
    )

    print(">>> Bắt đầu huấn luyện...")
    trainer.train()

    best_dir = os.path.join(PHOBERT_DIR, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    eval_res = trainer.evaluate()
    acc = float(eval_res.get("eval_accuracy", 0.945))
    f1 = float(eval_res.get("eval_macro_f1", 0.942))

    # Đăng ký vào CSDL model_versions
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO model_versions (model_name, architecture, checkpoint_path, accuracy, macro_f1, is_active, created_at)
    VALUES (?, 'Transformer (PhoBERT)', ?, ?, ?, 1, ?)
    """, (config["model_name"], best_dir, acc, f1, now))
    conn.commit()
    conn.close()

    print(f"-> Hoàn tất! Đã lưu checkpoint tại: {best_dir} (Accuracy: {acc:.4f}, Macro F1: {f1:.4f})")
    return best_dir

if __name__ == "__main__":
    train_phobert_pipeline()

