"""
Module Đánh giá mô hình trên tập Kiểm thử Độc lập (Test Set) & Phân tích lỗi (Error Analysis).
- Chạy 1 lần duy nhất trên tập Test để báo cáo kết quả cuối cùng.
- Xuất ma trận nhầm lẫn (Confusion Matrix).
- Lọc và phân tích các trường hợp mô hình đoán sai (False Positive / False Negative).
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score

def evaluate_model(
    model_dir: str = "outputs/phobert/best",
    test_path: str = "data/processed/test.csv",
    output_dir: str = "outputs/evaluation"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị kiểm thử: {device}")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Không tìm thấy file: {test_path}")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Không tìm thấy checkpoint mô hình tại: {model_dir}")

    print(f"[*] Đang tải mô hình & Tokenizer từ: {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    test_df = pd.read_csv(test_path)
    text_col = "text_segmented" if "text_segmented" in test_df.columns else ("text_seg" if "text_seg" in test_df.columns else "text")
    texts = test_df[text_col].astype(str).tolist()
    y_true = test_df["label"].astype(int).values

    print(f"[*] Đang dự đoán trên {len(texts)} mẫu của tập Test...")
    all_preds = []
    all_probs = []

    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # Xác suất nhãn 1 (Tin giả)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 1. Tính toán các chỉ số
    acc = accuracy_score(y_true, all_preds)
    macro_f1 = f1_score(y_true, all_preds, average="macro")
    try:
        auc = roc_auc_score(y_true, all_probs)
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, all_preds)
    report = classification_report(
        y_true, all_preds, target_names=["Tin Thật (0)", "Tin Giả (1)"], digits=4
    )

    print("\n" + "=" * 65)
    print(">>> BẢNG KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP <<<")
    print("=" * 65)
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nChi tiết Precision, Recall, F1 theo từng nhãn:")
    print(report)
    print("Confusion Matrix (Hàng: Thực tế, Cột: Dự đoán):")
    print(cm)
    print("=" * 65)

    # 2. Phân tích lỗi (Error Analysis)
    test_df["prediction"] = all_preds
    test_df["fake_probability"] = np.round(all_probs, 4)
    
    # Lỗi False Positive: Nhãn thật (0) nhưng đoán là giả (1)
    fp_df = test_df[(test_df["label"] == 0) & (test_df["prediction"] == 1)]
    # Lỗi False Negative: Nhãn giả (1) nhưng đoán là thật (0) -> rất nguy hiểm trong thực tế
    fn_df = test_df[(test_df["label"] == 1) & (test_df["prediction"] == 0)]

    print(f"\n[!] Thống kê sai sót (Errors):")
    print(f" - False Positives (Tin thật bị gán nhãn nhầm là giả): {len(fp_df)} mẫu")
    print(f" - False Negatives (Tin giả bị lọt lưới thành tin thật): {len(fn_df)} mẫu")

    # Lưu kết quả phân tích lỗi để viết báo cáo
    error_cases = test_df[test_df["label"] != test_df["prediction"]].copy()
    error_file = os.path.join(output_dir, "error_analysis.csv")
    error_cases.to_csv(error_file, index=False, encoding="utf-8")
    print(f"-> Đã lưu các mẫu dự đoán sai vào: {error_file}")

    # Lưu bảng tổng kết
    with open(os.path.join(output_dir, "metrics_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\nMacro F1: {macro_f1:.4f}\nROC-AUC: {auc:.4f}\n\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm}\n")

if __name__ == "__main__":
    evaluate_model()
