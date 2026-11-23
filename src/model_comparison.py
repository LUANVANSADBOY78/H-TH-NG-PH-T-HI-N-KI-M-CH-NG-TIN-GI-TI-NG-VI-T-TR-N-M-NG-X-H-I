"""
Module So sánh Nhiều Mô hình (Multi-Model Comparison Engine).
Đọc kết quả thực nghiệm thực tế từ các file metrics JSON (không hard-code).
- outputs/evaluation/test_metrics.json (PhoBERT-base-v2)
- outputs/evaluation/baseline_metrics.json (Logistic Regression, Linear SVM, Multinomial NB, Random Forest)
"""

import os
import sys
import json
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import EVALUATION_DIR

def get_comparison_summary() -> pd.DataFrame:
    """Trả về bảng tổng hợp so sánh các mô hình được sinh hoàn toàn từ thực nghiệm."""
    rows = []

    # 1. Nạp kết quả Baselines
    baseline_json = os.path.join(EVALUATION_DIR, "baseline_metrics.json")
    if os.path.exists(baseline_json):
        try:
            with open(baseline_json, "r", encoding="utf-8") as f:
                b_data = json.load(f)
            for m_name, vals in b_data.items():
                rows.append({
                    "Nhóm": "Truyền thống",
                    "Mô hình": m_name,
                    "Accuracy": vals.get("accuracy", 0.0),
                    "Precision": vals.get("precision", 0.0),
                    "Recall": vals.get("recall", 0.0),
                    "Macro F1": vals.get("macro_f1", 0.0),
                    "Nguồn số liệu": "Thực nghiệm thật (Test Set N=390)"
                })
        except Exception as e:
            print(f"[!] Lỗi đọc baseline_metrics.json: {e}")

    # 2. Nạp kết quả PhoBERT
    phobert_json = os.path.join(EVALUATION_DIR, "test_metrics.json")
    if os.path.exists(phobert_json):
        try:
            with open(phobert_json, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            m = p_data.get("metrics", {})
            rows.append({
                "Nhóm": "Transformer (SOTA)",
                "Mô hình": p_data.get("model_name", "PhoBERT-base-v2 (Thuần Văn Bản)"),
                "Accuracy": m.get("accuracy", 0.0),
                "Precision": m.get("macro_precision", 0.0),
                "Recall": m.get("macro_recall", 0.0),
                "Macro F1": m.get("macro_f1", 0.0),
                "Nguồn số liệu": "Thực nghiệm thật (Test Set N=390)"
            })
        except Exception as e:
            print(f"[!] Lỗi đọc test_metrics.json: {e}")

    if not rows:
        return pd.DataFrame([
            {"Nhóm": "Chưa có", "Mô hình": "Vui lòng chạy baseline.py và evaluate.py", "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "Macro F1": 0.0, "Nguồn số liệu": "N/A"}
        ])

    df = pd.DataFrame(rows)
    # Sắp xếp theo Macro F1 giảm dần
    df = df.sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
    return df

if __name__ == "__main__":
    print(get_comparison_summary().to_string(index=False))
