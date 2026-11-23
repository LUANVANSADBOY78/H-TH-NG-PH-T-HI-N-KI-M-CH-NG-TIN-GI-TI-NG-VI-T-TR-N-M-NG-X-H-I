"""
Script Xây Dựng & Hợp Nhất Bộ Dữ Liệu Phát Hiện Tin Giả Tiếng Việt Toàn Diện (Thuần Văn Bản):
- Nguồn 1: Benchmark ViFN Ver 3 (Vietnamese Fake News Dataset - Viện Hàn lâm KH&CN).
- Nguồn 2: Dữ liệu mạng xã hội thực tế (social_media_posts_raw.csv).
- Toàn bộ dữ liệu huấn luyện, kiểm định và kiểm thử tập trung 100% vào ngữ nghĩa văn bản.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime

# Đảm bảo mã hóa UTF-8 trên Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR
from src.database.connection import get_connection
from src.dataset.clean_data import clean_text, segment_text
from sklearn.model_selection import train_test_split

def build_dataset():
    print("=" * 70)
    print("🚀 BẮT ĐẦU QUÁ TRÌNH HỢP NHẤT BỘ DỮ LIỆU TIN GIẢ TIẾNG VIỆT (TEXT-ONLY)")
    print("=" * 70)

    # 1. Nguồn Mạng Xã Hội Thực tế (Social Media Raw)
    social_path = os.path.join(EXTERNAL_DATA_DIR, "social_media_posts_raw.csv")
    if os.path.exists(social_path):
        social_df = pd.read_csv(social_path, header=None, names=["text", "label"])
        social_df["source"] = "Mạng Xã Hội (Facebook/TikTok/Zalo)"
        print(f"[*] Đã tải {len(social_df)} bài viết mạng xã hội thực tế.")
    else:
        social_df = pd.DataFrame(columns=["text", "label", "source"])

    # 2. Nguồn Benchmark ViFN Ver 3
    vifn_dir = os.path.join(EXTERNAL_DATA_DIR, "ViFN", "ViFN-Vietnamese_Fake_New_Datasets_Ver3-main", "processed")
    vifn_fake_path = os.path.join(vifn_dir, "deduplicated_articles_fake.csv")
    vifn_real_path = os.path.join(vifn_dir, "deduplicated_articles_real.csv")

    def make_vifn_text(row):
        t = str(row["title"]).strip() if pd.notna(row["title"]) else ""
        s = str(row["summary"]).strip() if pd.notna(row["summary"]) else ""
        if t and s and s != t:
            return f"{t}: {s}"
        return t or s

    if os.path.exists(vifn_fake_path) and os.path.exists(vifn_real_path):
        vifn_fake = pd.read_csv(vifn_fake_path)
        vifn_real = pd.read_csv(vifn_real_path)

        vifn_fake["text"] = vifn_fake.apply(make_vifn_text, axis=1)
        vifn_fake["label"] = 1
        vifn_fake["source"] = "ViFN Dataset (Tin Giả)"

        vifn_real["text"] = vifn_real.apply(make_vifn_text, axis=1)
        vifn_real["label"] = 0
        vifn_real["source"] = "ViFN Dataset (Tin Thật)"

        vifn_real_sample = vifn_real.sample(n=min(1500, len(vifn_real)), random_state=42)
        print(f"[*] Đã tải {len(vifn_fake)} tin giả và {len(vifn_real_sample)} tin thật từ ViFN Ver 3.")
    else:
        vifn_fake = pd.DataFrame(columns=["text", "label", "source"])
        vifn_real_sample = pd.DataFrame(columns=["text", "label", "source"])

    # 3. Kết hợp toàn bộ các nguồn
    combined = pd.concat([
        social_df[["text", "label", "source"]],
        vifn_fake[["text", "label", "source"]],
        vifn_real_sample[["text", "label", "source"]]
    ], ignore_index=True)

    # 4. Làm sạch & Lọc nhiễu
    combined = combined.dropna(subset=["text", "label"])
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 15]
    combined = combined.drop_duplicates(subset=["text"])
    combined["label"] = combined["label"].astype(int)

    total_samples = len(combined)
    fake_count = int((combined["label"] == 1).sum())
    real_count = int((combined["label"] == 0).sum())
    print(f"\n[+] Tổng số bài viết sau khi khử trùng lặp: {total_samples}")
    print(f"    - Tin Thật (Label 0): {real_count} ({real_count / total_samples * 100:.1f}%)")
    print(f"    - Tin Giả  (Label 1): {fake_count} ({fake_count / total_samples * 100:.1f}%)")

    # 5. Gán ID bài viết
    ids = []
    for idx, row in combined.iterrows():
        lbl = row["label"]
        src = "social" if "Mạng Xã Hội" in str(row["source"]) else "vifn"
        prefix = f"{src}_{'fake' if lbl == 1 else 'real'}"
        ids.append(f"{prefix}_{idx:05d}")
    combined["id"] = ids

    # 6. Tiền Xử Lý Ngôn Ngữ (Clean & Word Segmentation)
    print("\n[*] Đang tiến hành làm sạch và tách từ tiếng Việt (Underthesea)...")
    combined["text_clean"] = combined["text"].apply(clean_text)
    combined["text_segmented"] = combined["text_clean"].apply(segment_text)
    combined["text_seg"] = combined["text_segmented"]

    # Lưu tập raw tổng hợp
    raw_combined_path = os.path.join(RAW_DATA_DIR, "data.csv")
    combined.to_csv(raw_combined_path, index=False, encoding="utf-8")
    print(f"[✔] Đã lưu dữ liệu tổng hợp tại: {raw_combined_path}")

    # 7. Phân chia phân tầng chuẩn (Stratified Split 70% Train / 15% Val / 15% Test)
    train_val, test_df = train_test_split(combined, test_size=0.15, stratify=combined["label"], random_state=42)
    train_df, val_df = train_test_split(train_val, test_size=0.15 / (1.0 - 0.15), stratify=train_val["label"], random_state=42)

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DATA_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")

    cols_to_save = ["id", "text", "text_clean", "text_segmented", "text_seg", "label", "source"]
    train_df[cols_to_save].to_csv(train_path, index=False, encoding="utf-8")
    val_df[cols_to_save].to_csv(val_path, index=False, encoding="utf-8")
    test_df[cols_to_save].to_csv(test_path, index=False, encoding="utf-8")

    print(f"\n[✔] Tập Huấn luyện (Train): {len(train_df)} mẫu -> {train_path}")
    print(f"[✔] Tập Kiểm định  (Val):   {len(val_df)} mẫu -> {val_path}")
    print(f"[✔] Tập Kiểm thử   (Test):  {len(test_df)} mẫu -> {test_path}")

    # Đăng ký vào SQLite
    try:
        conn = get_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        INSERT INTO datasets (name, file_path, row_count, split_ratio, version, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "ViFN Ver3 & Social Media Dataset (Text-Only Clean)",
            raw_combined_path,
            total_samples,
            "70/15/15",
            "v3.1-TextOnly",
            now
        ))
        conn.commit()
        conn.close()
        print("[✔] Đã cập nhật metadata vào SQLite Database.")
    except Exception as e:
        print(f"[!] SQLite register notice: {e}")

    print("\n✅ HOÀN TẤT XÂY DỰNG DỮ LIỆU SẠCH (KHÔNG RÒ RỈ METADATA).")

if __name__ == "__main__":
    build_dataset()
