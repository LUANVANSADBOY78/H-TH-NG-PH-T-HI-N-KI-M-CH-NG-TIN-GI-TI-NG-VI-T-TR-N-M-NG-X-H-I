"""
Kịch bản Huấn luyện (Fine-tuning) mô hình Transformer (PhoBERT-base-v2).
Hỗ trợ cả 2 chế độ:
1. Chế độ Thuần Văn Bản (Pure Text): Sử dụng Hugging Face Trainer tiêu chuẩn.
2. Chế độ Mở rộng (Text + Metadata): Sử dụng PhoBERTWithMetadata để khai thác số liệu like/share/cmt.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
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

def compute_metrics(eval_pred):
    """Tính toán Accuracy, Precision, Recall và Macro F1."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "macro_f1": round(f1, 4)
    }

def train_phobert(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị huấn luyện (Device): {device}")
    if torch.cuda.is_available():
        print(f"[*] Tên GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    # 1. Tải Tokenizer PhoBERT
    print(f"[*] Đang tải Tokenizer cho {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    # 2. Đọc dữ liệu đã tiền xử lý
    train_path = os.path.join(args.data_dir, "train.csv")
    val_path = os.path.join(args.data_dir, "val.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Không tìm thấy tập dữ liệu trong {args.data_dir}. Vui lòng chạy preprocess.py trước!")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Cột text đã tách từ
    text_col = "text_segmented" if "text_segmented" in train_df.columns else ("text_seg" if "text_seg" in train_df.columns else "text")
    
    print(f"[*] Số mẫu Train: {len(train_df)} | Số mẫu Val: {len(val_df)}")
    print(f"[*] Cột văn bản sử dụng: '{text_col}'")

    train_ds = Dataset.from_pandas(train_df[[text_col, "label"]].rename(columns={text_col: "text"}))
    val_ds = Dataset.from_pandas(val_df[[text_col, "label"]].rename(columns={text_col: "text"}))

    # Tokenize dữ liệu với dynamic padding
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    print("[*] Đang mã hóa dữ liệu văn bản (Tokenizing)...")
    train_tokenized = train_ds.map(tokenize_fn, batched=True)
    val_tokenized = val_ds.map(tokenize_fn, batched=True)

    # 3. Khởi tạo mô hình
    print(f"[*] Đang tải mô hình tiền huấn luyện: {args.model_name} ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels
    )

    # 4. Thiết lập siêu tham số huấn luyện (TrainingArguments tối ưu cho RTX 3050 4GB)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,       # Tránh tràn 4GB VRAM
        gradient_accumulation_steps=args.gradient_accumulation_steps, # Giữ effective batch size mà nhẹ VRAM
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        fp16=torch.cuda.is_available(),                   # Bật fp16 mixed precision tăng tốc 3x
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=10,
        dataloader_num_workers=0,                         # Tránh deadlock treo máy trên Windows
        optim="adamw_torch",
        seed=args.seed,
        report_to="none"
    )

    # 5. Khởi tạo Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    # 6. Bắt đầu Huấn luyện
    print("\n" + "=" * 60)
    print(">>> BẮT ĐẦU FINE-TUNING PHOBERT <<<")
    print("=" * 60)
    train_result = trainer.train()

    # 7. Lưu mô hình và Tokenizer tốt nhất
    best_model_dir = os.path.join(args.output_dir, "best")
    os.makedirs(best_model_dir, exist_ok=True)
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    print("\n" + "=" * 60)
    print(f">>> HUẤN LUYỆN HOÀN TẤT! Mô hình tốt nhất đã lưu tại: {best_model_dir} <<<")
    print("=" * 60)

    # Đánh giá lại trên tập val
    val_eval = trainer.evaluate()
    print("\n--- KẾT QUẢ CUỐI CÙNG TRÊN TẬP VALIDATION ---")
    for k, v in val_eval.items():
        if "eval_" in k:
            print(f"{k.replace('eval_', '').upper():<15}: {v}")

    return trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT cho bài toán phát hiện tin giả tiếng Việt.")
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base-v2", help="Mã mô hình HuggingFace")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Thư mục chứa train.csv, val.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/phobert", help="Thư mục lưu checkpoint")
    parser.add_argument("--max_length", type=int, default=192, help="Độ dài tối đa của câu sau khi tokenize")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size huấn luyện (8 tối ưu cho RTX 3050 4GB)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Số bước tích lũy gradient")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Tốc độ học (Learning rate)")
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch huấn luyện tối đa")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Hệ số suy giảm trọng số L2")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Số bước warmup của learning rate scheduler")
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="Số epoch chờ trước khi dừng sớm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed để tái lập kết quả")
    parser.add_argument("--num_labels", type=int, default=2, help="Số lớp (2: Thật/Giả)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_phobert(args)
