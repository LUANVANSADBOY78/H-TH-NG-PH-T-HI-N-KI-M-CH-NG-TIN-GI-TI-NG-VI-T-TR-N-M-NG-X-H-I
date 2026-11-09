"""
Module Kiểm Tra & Đánh Giá Chất Lượng Tập Dữ Liệu (Data Quality Validation).
Kiểm tra:
- Số lượng bản ghi trùng lặp (Duplicates)
- Số lượng giá trị thiếu (Null values)
- Tỉ lệ mất cân bằng lớp (Class Imbalance Ratio)
- Phân bố độ dài từ (Outliers)
"""

import pandas as pd
import numpy as np

def validate_dataset(file_path: str) -> dict:
    df = pd.read_csv(file_path)
    total_rows = len(df)
    
    # 1. Null check
    null_texts = int(df["text"].isna().sum()) if "text" in df.columns else 0
    null_labels = int(df["label"].isna().sum()) if "label" in df.columns else 0

    # 2. Duplicate check
    duplicate_texts = int(df.duplicated(subset=["text"]).sum()) if "text" in df.columns else 0

    # 3. Class balance
    class_counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}

    # 4. Length stats
    if "text" in df.columns:
        lengths = df["text"].astype(str).apply(lambda x: len(x.split()))
        avg_len = float(lengths.mean())
        p95_len = float(lengths.quantile(0.95))
        short_texts = int((lengths < 5).sum())
    else:
        avg_len = p95_len = short_texts = 0

    # Đánh giá chung
    issues = []
    if duplicate_texts > 0:
        issues.append(f"Phát hiện {duplicate_texts} bản ghi trùng lặp nội dung.")
    if null_texts > 0:
        issues.append(f"Có {null_texts} dòng văn bản bị rỗng (null).")
    if short_texts > 0:
        issues.append(f"Có {short_texts} bài viết quá ngắn (<5 từ).")

    is_healthy = len(issues) == 0

    return {
        "total_rows": total_rows,
        "null_count": null_texts + null_labels,
        "duplicate_count": duplicate_texts,
        "class_distribution": class_counts,
        "avg_length": round(avg_len, 1),
        "p95_length": round(p95_len, 1),
        "short_samples_count": short_texts,
        "is_healthy": is_healthy,
        "issues": issues
    }

if __name__ == "__main__":
    import os
    sample = "data/raw/data.csv"
    if os.path.exists(sample):
        print(validate_dataset(sample))

