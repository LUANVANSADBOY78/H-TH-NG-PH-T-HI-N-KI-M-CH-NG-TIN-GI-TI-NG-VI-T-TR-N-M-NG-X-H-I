"""
Module Đánh giá Năng lực Tổng Quát Hóa Đa Nguồn & Độ Vững Chắc (Cross-Domain & Robustness Evaluation).
Hai thực nghiệm chuyên sâu:
1. Đánh giá chéo nguồn (Cross-domain):
   - Huấn luyện trên Nguồn A (Báo chí chính thống - ViFN Ver 3)
   - Kiểm tra trên Nguồn B (Bài đăng Mạng Xã Hội - Facebook/TikTok/Zalo)
   - Đo lường mức độ suy giảm hiệu năng ngoài phân phối (Out-of-Distribution Degradation).
2. Kiểm tra che tên báo & URL (Source Entity & URL Masking):
   - Thay thế toàn bộ liên kết web, tên tờ báo (VnExpress, Tuổi Trẻ, Zing, Dân Trí...) bằng [NGUỒN_ĐÃ_CHE]
   - Kiểm chứng xem mô hình có thực sự hiểu ngữ nghĩa văn bản hay chỉ học vẹt 'thương hiệu nguồn tin'.
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, EVALUATION_DIR

def mask_source_and_urls(text: str) -> str:
    """Che toàn bộ URL và tên các cơ quan báo chí / nguồn tin."""
    t = str(text)
    # 1. Che URL
    t = re.sub(r"https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.(?:vn|com|org|net|edu)\b", " [LIEN_KET_WEB] ", t, flags=re.IGNORECASE)
    # 2. Che cụm dẫn nguồn phổ biến
    news_sources = [
        r"\b(?:theo\s+)?vnexpress(?:\.net)?\b",
        r"\b(?:theo\s+)?tuổi\s+trẻ(?:\s+online)?\b",
        r"\b(?:theo\s+)?thanh\s+niên\b",
        r"\b(?:theo\s+)?dân\s+trí\b",
        r"\b(?:theo\s+)?zing(?:\s+news)?\b",
        r"\b(?:theo\s+)?vtv(?:\s+news|\s+1|\s+24)?\b",
        r"\b(?:theo\s+)?vietnamnet\b",
        r"\b(?:theo\s+)?lao\s+động\b",
        r"\b(?:theo\s+)?công\s+an\s+nhân\s+dân\b",
        r"\b(?:theo\s+)?bộ\s+y\s+tế\b",
        r"\b(?:theo\s+)?chính\s+phủ\b",
        r"\b(?:theo\s+)?bộ\s+giáo\s+dục\b"
    ]
    for pattern in news_sources:
        t = re.sub(pattern, " [NGUON_DA_CHE] ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def run_cross_domain_evaluation():
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_DATA_DIR, "data.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Chưa có dữ liệu tại {raw_path}")

    df = pd.read_csv(raw_path)
    text_col = "text_segmented" if "text_segmented" in df.columns else ("text_seg" if "text_seg" in df.columns else "text")

    # Phân loại 2 nguồn dữ liệu
    is_social = df["source"].astype(str).str.contains("Mạng Xã Hội|social", case=False)
    df_vifn = df[~is_social].copy()
    df_social = df[is_social].copy()

    print("\n" + "=" * 68)
    print(">>> 1. THỰC NGHIỆM ĐÁNH GIÁ CHÉO NGUỒN (CROSS-DOMAIN EVALUATION) <<<")
    print("=" * 68)
    print(f"[*] Nguồn A (ViFN - Báo chí chính thống): {len(df_vifn)} bài viết")
    print(f"[*] Nguồn B (Mạng xã hội - Facebook/TikTok): {len(df_social)} bài viết")

    # Huấn luyện trên ViFN -> Kiểm tra trên Mạng Xã Hội
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_vifn = vec.fit_transform(df_vifn[text_col].astype(str))
    y_vifn = df_vifn["label"].astype(int).values

    X_social = vec.transform(df_social[text_col].astype(str))
    y_social = df_social["label"].astype(int).values

    # In-domain ViFN split
    from sklearn.model_selection import train_test_split
    X_train_v, X_val_v, y_train_v, y_val_v = train_test_split(X_vifn, y_vifn, test_size=0.2, random_state=42, stratify=y_vifn)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_train_v, y_train_v)

    # Đánh giá trong miền (In-domain)
    preds_in = clf.predict(X_val_v)
    acc_in = accuracy_score(y_val_v, preds_in)
    f1_in = f1_score(y_val_v, preds_in, average="macro")

    # Đánh giá ngoài miền (Cross-domain sang mạng xã hội)
    preds_cross = clf.predict(X_social)
    acc_cross = accuracy_score(y_social, preds_cross)
    f1_cross = f1_score(y_social, preds_cross, average="macro")

    print(f"\n[+] Kết quả trong miền (Train trên ViFN -> Test trên ViFN):")
    print(f"    - Accuracy: {acc_in:.4f} | Macro F1: {f1_in:.4f}")
    print(f"[+] Kết quả chéo miền (Train trên ViFN -> Test trên Mạng Xã Hội):")
    print(f"    - Accuracy: {acc_cross:.4f} | Macro F1: {f1_cross:.4f}")
    delta_f1 = (f1_cross - f1_in) * 100
    print(f"    - Độ dịch chuyển hiệu năng (Domain Shift): {delta_f1:+.2f}%")

    print("\n" + "=" * 68)
    print(">>> 2. THỰC NGHIỆM CHE TÊN BÁO & URL (SOURCE ENTITY MASKING) <<<")
    print("=" * 68)
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")
    test_df = pd.read_csv(test_path)
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    train_df = pd.read_csv(train_path)

    # Huấn luyện trên văn bản gốc
    vec_orig = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    X_tr_orig = vec_orig.fit_transform(train_df[text_col].astype(str))
    y_tr = train_df["label"].astype(int).values

    clf_mask = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf_mask.fit(X_tr_orig, y_tr)

    # Đánh giá trên test gốc
    X_te_orig = vec_orig.transform(test_df[text_col].astype(str))
    y_te = test_df["label"].astype(int).values
    preds_orig = clf_mask.predict(X_te_orig)
    acc_orig = accuracy_score(y_te, preds_orig)
    f1_orig = f1_score(y_te, preds_orig, average="macro")

    # Che tên báo và liên kết trong tập test
    masked_test_texts = test_df[text_col].astype(str).apply(mask_source_and_urls)
    X_te_masked = vec_orig.transform(masked_test_texts)
    preds_masked = clf_mask.predict(X_te_masked)
    acc_masked = accuracy_score(y_te, preds_masked)
    f1_masked = f1_score(y_te, preds_masked, average="macro")

    print(f"[*] Tập Test gốc:    Accuracy = {acc_orig:.4f} | Macro F1 = {f1_orig:.4f}")
    print(f"[*] Tập Test đã che: Accuracy = {acc_masked:.4f} | Macro F1 = {f1_masked:.4f}")
    mask_impact = (f1_masked - f1_orig) * 100
    print(f"[*] Mức độ ảnh hưởng khi che tên nguồn tin: {mask_impact:+.2f}%")

    robustness_data = {
        "cross_domain_evaluation": {
            "source_a_vifn_count": len(df_vifn),
            "source_b_social_count": len(df_social),
            "in_domain_vifn": {
                "accuracy": round(float(acc_in), 4),
                "macro_f1": round(float(f1_in), 4)
            },
            "cross_domain_social": {
                "accuracy": round(float(acc_cross), 4),
                "macro_f1": round(float(f1_cross), 4)
            },
            "performance_drop_f1_percentage": round(float(delta_f1), 2)
        },
        "source_name_and_url_masking": {
            "original_test": {
                "accuracy": round(float(acc_orig), 4),
                "macro_f1": round(float(f1_orig), 4)
            },
            "masked_test": {
                "accuracy": round(float(acc_masked), 4),
                "macro_f1": round(float(f1_masked), 4)
            },
            "impact_f1_percentage": round(float(mask_impact), 2)
        }
    }

    out_file = os.path.join(EVALUATION_DIR, "robustness_metrics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(robustness_data, f, indent=2, ensure_ascii=False)
    print(f"\n[✔] Đã lưu kết quả thực nghiệm Robustness tại: {out_file}")

    return robustness_data

if __name__ == "__main__":
    run_cross_domain_evaluation()
