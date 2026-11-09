"""
Phân tích khám phá dữ liệu (EDA - Exploratory Data Analysis).
Xuất các thông số phân bố nhãn, độ dài văn bản và top từ khóa đặc trưng.
"""

import os
import sys
import pandas as pd
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def perform_eda(data_path: str = "data/raw/data.csv"):
    if not os.path.exists(data_path):
        print(f"Chưa tìm thấy {data_path}. Vui lòng tạo hoặc đặt dữ liệu vào thư mục.")
        return

    df = pd.read_csv(data_path)
    print("=" * 60)
    print(f"--- BÁO CÁO PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA): {data_path} ---")
    print("=" * 60)

    # 1. Thống kê số lượng & tỉ lệ nhãn
    label_counts = df["label"].value_counts()
    print("\n1. Phân bố nhãn (0: Tin thật, 1: Tin giả):")
    for lbl, count in label_counts.items():
        name = "Tin Thật" if lbl == 0 else "Tin Giả"
        print(f" - Lớp {lbl} ({name}): {count} mẫu ({count/len(df)*100:.1f}%)")

    # 2. Thống kê độ dài văn bản (Số từ)
    df["word_count"] = df["text"].astype(str).apply(lambda x: len(x.split()))
    print("\n2. Thống kê độ dài văn bản (theo số lượng từ):")
    print(f" - Độ dài trung bình : {df['word_count'].mean():.1f} từ")
    print(f" - Độ dài ngắn nhất  : {df['word_count'].min()} từ")
    print(f" - Độ dài dài nhất   : {df['word_count'].max()} từ")
    print(f" - Phân vị 90% (P90) : {df['word_count'].quantile(0.9):.1f} từ")
    print(f" - Phân vị 95% (P95) : {df['word_count'].quantile(0.95):.1f} từ")
    print(" -> Khuyến nghị: Chọn max_length = 128 hoặc 256 cho PhoBERT là bao phủ được >95% văn bản.")

    # 3. Thống kê tương tác mạng xã hội (Metadata) nếu có
    if "likes" in df.columns and "shares" in df.columns:
        print("\n3. Thống kê tương tác mạng xã hội trung bình theo từng lớp:")
        meta_stats = df.groupby("label")[["likes", "shares", "comments"]].mean().round(1)
        print(meta_stats)

    print("=" * 60)

if __name__ == "__main__":
    perform_eda()

