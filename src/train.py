"""
Kịch bản Huấn luyện (Fine-tuning) mô hình Transformer PhoBERT-base-v2 (Thuần Văn Bản).
- Tối ưu hóa cho GPU máy trạm / Laptop (NVIDIA RTX 3050 4GB VRAM).
- Sử dụng Weighted CrossEntropyLoss để xử lý mất cân bằng lớp (Tin Thật: 269 vs Tin Giả: 121).
- Tự động lưu checkpoint tốt nhất vào outputs/phobert/best/ và ghi nhận training_history.json.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    DEFAULT_MODEL_NAME, MAX_LENGTH, BATCH_SIZE, LEARNING_RATE,
    EPOCHS, WEIGHT_DECAY, SEED, PHOBERT_OUTPUT_DIR, PHOBERT_BEST_DIR,
    PROCESSED_DATA_DIR, CLASS_WEIGHTS
)

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

class WeightedLossTrainer(Trainer):
    """Trainer tùy biến có tích hợp trọng số lớp (Class Weights) vào hàm mất mát."""
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, dtype=torch.float).to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    """Tính toán Accuracy, Precision, Recall và Macro F1."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4)
    }

def train_phobert(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị huấn luyện (Device): {device}")
    if torch.cuda.is_available():
        print(f"[*] Tên GPU: {torch.cuda.get_device_name(0)}")
        print(f"[*] Bộ nhớ VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB")
        torch.cuda.empty_cache()

    # 1. Tải Tokenizer PhoBERT
    print(f"[*] Đang tải Tokenizer cho {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    # 2. Đọc dữ liệu đã tiền xử lý
    train_path = os.path.join(args.data_dir, "train.csv")
    val_path = os.path.join(args.data_dir, "val.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Không tìm thấy tập dữ liệu trong {args.data_dir}. Vui lòng chạy build_complete_dataset.py trước!")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    text_col = "text_segmented" if "text_segmented" in train_df.columns else ("text_seg" if "text_seg" in train_df.columns else "text")
    print(f"[*] Số mẫu Train: {len(train_df)} | Số mẫu Val: {len(val_df)}")
    print(f"[*] Cột văn bản sử dụng: '{text_col}'")

    train_ds = Dataset.from_pandas(train_df[[text_col, "label"]].rename(columns={text_col: "text"}))
    val_ds = Dataset.from_pandas(val_df[[text_col, "label"]].rename(columns={text_col: "text"}))

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    print("[*] Đang mã hóa dữ liệu văn bản (Tokenizing)...")
    train_tokenized = train_ds.map(tokenize_fn, batched=True)
    val_tokenized = val_ds.map(tokenize_fn, batched=True)

    # 3. Khởi tạo mô hình
    print(f"[*] Đang tải mô hình tiền huấn luyện: {args.model_name} (num_labels={args.num_labels}) ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels
    )

    # Tính toán Class Weights nếu có mất cân bằng
    n_real = int((train_df["label"] == 0).sum())
    n_fake = int((train_df["label"] == 1).sum())
    w_real = 1.0
    w_fake = round(n_real / max(n_fake, 1), 3)
    dynamic_weights = [w_real, w_fake]
    print(f"[*] Trọng số lớp phân bổ (Class Weights): Real={w_real}, Fake={w_fake} (tỷ lệ {n_real}:{n_fake})")

    # 4. Thiết lập siêu tham số huấn luyện
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=20,
        dataloader_num_workers=0,
        optim="adamw_torch",
        seed=args.seed,
        report_to="none"
    )

    trainer = WeightedLossTrainer(
        class_weights=dynamic_weights,
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    # 5. Huấn luyện
    print("\n" + "=" * 65)
    print(">>> BẮT ĐẦU FINE-TUNING PHOBERT-BASE-V2 TRÊN DỮ LIỆU THẬT <<<")
    print("=" * 65)
    train_result = trainer.train()

    # 6. Lưu mô hình và Tokenizer tốt nhất
    best_model_dir = os.path.join(args.output_dir, "best")
    os.makedirs(best_model_dir, exist_ok=True)
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    print("\n" + "=" * 65)
    print(f">>> HUẤN LUYỆN HOÀN TẤT! Checkpoint tốt nhất đã lưu tại: {best_model_dir} <<<")
    print("=" * 65)

    # 7. Trích xuất và lưu lịch sử huấn luyện
    history_file = os.path.join(args.output_dir, "training_history.json")
    history_data = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "class_weights": dynamic_weights,
        "log_history": trainer.state.log_history
    }
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)
    print(f"[*] Đã lưu lịch sử huấn luyện (Training History) tại: {history_file}")

    return trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT cho bài toán phát hiện tin giả tiếng Việt.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Mã mô hình HuggingFace")
    parser.add_argument("--data_dir", type=str, default=PROCESSED_DATA_DIR, help="Thư mục chứa train.csv, val.csv")
    parser.add_argument("--output_dir", type=str, default=PHOBERT_OUTPUT_DIR, help="Thư mục lưu checkpoint")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Độ dài tối đa của câu sau khi tokenize")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size huấn luyện")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Số bước tích lũy gradient")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Tốc độ học")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Số epoch huấn luyện")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Hệ số suy giảm trọng số L2")
    parser.add_argument("--warmup_steps", type=int, default=30, help="Số bước warmup")
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="Patience dừng sớm")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--num_labels", type=int, default=2, help="Số lớp (2: Thật/Giả)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_phobert(args)
