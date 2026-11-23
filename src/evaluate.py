"""
Module Đánh giá mô hình trên tập Kiểm thử Độc lập (Test Set) & Phân tích lỗi (Error Analysis).
- Đánh giá khách quan 1 lần duy nhất trên 390 mẫu của tập Test.
- Xuất số liệu chuẩn xác ra test_metrics.json (không hardcode).
- Vẽ ma trận nhầm lẫn thực nghiệm và lưu biểu đồ ra confusion_matrix.png.
- Trích xuất bảng phân tích các ca dự đoán sai (error_analysis.csv).
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PHOBERT_BEST_DIR, EVALUATION_DIR, PROCESSED_DATA_DIR, MAX_LENGTH, OUTPUTS_DIR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_recall_fscore_support

def evaluate_model(
    model_dir: str = PHOBERT_BEST_DIR,
    test_path: str = os.path.join(PROCESSED_DATA_DIR, "test.csv"),
    output_dir: str = EVALUATION_DIR
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị kiểm thử (Inference Device): {device}")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Không tìm thấy file tập test tại: {test_path}")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Không tìm thấy checkpoint mô hình tại: {model_dir}. Vui lòng chạy train.py trước!")

    print(f"[*] Đang nạp mô hình & Tokenizer từ: {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    test_df = pd.read_csv(test_path)
    text_col = "text_segmented" if "text_segmented" in test_df.columns else ("text_seg" if "text_seg" in test_df.columns else "text")
    texts = test_df[text_col].astype(str).tolist()
    y_true = test_df["label"].astype(int).values

    total_samples = len(texts)
    n_class_0 = int(np.sum(y_true == 0))
    n_class_1 = int(np.sum(y_true == 1))
    majority_acc = n_class_0 / total_samples

    print(f"[*] Đang suy luận trên tập Test: {total_samples} mẫu (Lớp 0: {n_class_0}, Lớp 1: {n_class_1})")
    print(f"[*] Baseline đa số (Predict all Real): {majority_acc:.6f} ({majority_acc * 100:.2f}%)")

    all_preds = []
    all_probs = []
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 1. Đo kiểm các chỉ số chuẩn xác
    acc = float(accuracy_score(y_true, all_preds))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, all_preds, average="macro", zero_division=0)
    prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(y_true, all_preds, average=None, zero_division=0)

    try:
        auc = float(roc_auc_score(y_true, all_probs))
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, all_preds)
    tn, fp, fn, tp = cm.ravel()

    report_str = classification_report(
        y_true, all_preds, target_names=["Tin Thật (0)", "Tin Giả (1)"], digits=4
    )

    print("\n" + "=" * 68)
    print(">>> KẾT QUẢ ĐÁNH GIÁ THỰC NGHIỆM TRÊN TẬP TEST ĐỘC LẬP (N=390) <<<")
    print("=" * 68)
    print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Macro F1 : {f1_macro:.4f}")
    print(f"Precision: {prec_macro:.4f}")
    print(f"Recall   : {rec_macro:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nChi tiết theo từng lớp:")
    print(report_str)
    print(f"Ma trận Nhầm lẫn (Confusion Matrix):")
    print(f" [[{tn:>4} (TN), {fp:>4} (FP)]")
    print(f"  [{fn:>4} (FN), {tp:>4} (TP)]]")
    print(f" -> Hàng 0 (Tin Thật): {tn} + {fp} = {tn + fp} (Ground Truth: {n_class_0})")
    print(f" -> Hàng 1 (Tin Giả ): {fn} + {tp} = {fn + tp} (Ground Truth: {n_class_1})")
    print("=" * 68)

    # 2. Lưu kết quả cấu trúc vào file JSON (Nguồn duy nhất cho báo cáo & UI)
    metrics_data = {
        "model_name": "PhoBERT-base-v2 (Thuần Văn Bản)",
        "total_test_samples": total_samples,
        "class_distribution": {
            "real_class_0": n_class_0,
            "fake_class_1": n_class_1
        },
        "majority_baseline_accuracy": round(majority_acc, 6),
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(float(f1_macro), 4),
            "macro_precision": round(float(prec_macro), 4),
            "macro_recall": round(float(rec_macro), 4),
            "roc_auc": round(float(auc), 4)
        },
        "per_class": {
            "real": {
                "precision": round(float(prec_per_class[0]), 4),
                "recall": round(float(rec_per_class[0]), 4),
                "f1": round(float(f1_per_class[0]), 4),
                "support": int(n_class_0)
            },
            "fake": {
                "precision": round(float(prec_per_class[1]), 4),
                "recall": round(float(rec_per_class[1]), 4),
                "f1": round(float(f1_per_class[1]), 4),
                "support": int(n_class_1)
            }
        },
        "confusion_matrix": {
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "true_negative_real_correct": int(tn),
            "false_positive_real_as_fake": int(fp),
            "false_negative_fake_missed": int(fn),
            "true_positive_fake_correct": int(tp),
            "row_sums_check": [int(tn + fp), int(fn + tp)]
        }
    }

    metrics_json_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print(f"[✔] Đã lưu metrics dạng JSON tại: {metrics_json_path}")

    # 3. Vẽ biểu đồ Confusion Matrix thật và lưu file ảnh
    plt.figure(figsize=(6.5, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Tin Thật (Real)', 'Tin Giả (Fake)'],
        yticklabels=['Tin Thật (Real)', 'Tin Giả (Fake)'],
        cbar=True, annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix (PhoBERT trên Test Set N={total_samples})", fontsize=11, fontweight='bold')
    plt.xlabel("Nhãn Dự Đoán (Predicted)", fontsize=10)
    plt.ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=10)
    plt.tight_layout()

    cm_img_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_img_path, dpi=200)
    # Copy sang outputs/ để tương thích với app.py cũ
    compat_cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(compat_cm_path, dpi=200)
    plt.close()
    print(f"[✔] Đã lưu biểu đồ ma trận thật tại: {cm_img_path}")

    # 4. Trích xuất phân tích lỗi (Error Analysis)
    test_df["prediction"] = all_preds
    test_df["fake_probability"] = np.round(all_probs, 4)
    error_cases = test_df[test_df["label"] != test_df["prediction"]].copy()
    error_file = os.path.join(output_dir, "error_analysis.csv")
    error_cases.to_csv(error_file, index=False, encoding="utf-8")
    print(f"[✔] Đã lưu {len(error_cases)} ca dự đoán sai để phân tích tại: {error_file}")

    return metrics_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mô hình PhoBERT trên tập test.")
    parser.add_argument("--model_dir", type=str, default=PHOBERT_BEST_DIR, help="Thư mục checkpoint mô hình")
    parser.add_argument("--test_path", type=str, default=os.path.join(PROCESSED_DATA_DIR, "test.csv"), help="File test.csv")
    parser.add_argument("--output_dir", type=str, default=EVALUATION_DIR, help="Thư mục lưu báo cáo")
    cli_args = parser.parse_args()

    evaluate_model(cli_args.model_dir, cli_args.test_path, cli_args.output_dir)
