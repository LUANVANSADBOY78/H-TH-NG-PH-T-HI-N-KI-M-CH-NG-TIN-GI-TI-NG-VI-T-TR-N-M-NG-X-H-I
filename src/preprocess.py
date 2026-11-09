"""
Module Tiền xử lý dữ liệu mạng xã hội tiếng Việt cho mô hình Transformer (PhoBERT).
Các bước thực hiện:
1. Chuẩn hóa bảng mã tiếng Việt sang Unicode dựng sẵn (NFC).
2. Làm sạch văn bản: loại bỏ liên kết URL, thẻ HTML, ký tự thừa, icon rác.
3. Chuẩn hóa teencode & tiếng lóng phổ biến trên mạng xã hội Việt Nam.
4. Xử lý Emoji (thay thế hoặc loại bỏ).
5. Tách từ tiếng Việt (Word Segmentation): Sinh viên -> Sinh_viên (bắt buộc với PhoBERT).
6. Phân chia tập dữ liệu Train / Val / Test theo tỉ lệ phân tầng (Stratified Split).
"""

import os
import sys
import re
import unicodedata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Từ điển ánh xạ teencode phổ biến
TEENCODE_DICT = {
    "ko": "không", "k": "không", "kh": "không", "hem": "không", "hông": "không",
    "dc": "được", "dk": "được", "đc": "được",
    "mn": "mọi người", "mng": "mọi người",
    "r": "rồi", "rùi": "rồi",
    "j": "gì", "gi": "gì",
    "wa": "quá", "wá": "quá",
    "thui": "thôi",
    "bit": "biết", "bik": "biết",
    "nt": "nhắn tin", "ib": "nhắn tin", "inbox": "nhắn tin",
    "cx": "cũng",
    "vs": "với", "zới": "với",
    "ng": "người", "ngừi": "người",
    "bn": "bạn", "b": "bạn",
    "e": "em", "a": "anh", "c": "chị",
    "cmt": "bình luận", "share": "chia sẻ", "fb": "facebook",
    "vcl": "rất", "vđ": "rất", "vl": "rất"
}

def normalize_unicode(text: str) -> str:
    """
    Chuẩn hóa chuỗi văn bản về chuẩn Unicode NFC dựng sẵn,
    tự động chuyển đổi các font chữ nghệ thuật trên mạng xã hội (Mathematical Bold, Italic) về chữ chuẩn.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return unicodedata.normalize("NFC", decomposed)

def replace_teencode(text: str) -> str:
    """Thay thế teencode và viết tắt dựa trên từ điển chuẩn."""
    words = text.split()
    normalized_words = [TEENCODE_DICT.get(w.lower(), w) for w in words]
    return " ".join(normalized_words)

def clean_social_text(text: str, remove_emoji: bool = True) -> str:
    """
    Làm sạch văn bản mạng xã hội:
    - Chuẩn hóa Unicode
    - Xóa URL (http/https/www)
    - Xóa tag HTML (<...>)
    - Xóa ký tự điều khiển
    - Thay thế teencode
    - Xử lý emoji
    """
    text = normalize_unicode(text)
    # Xóa URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Xóa HTML
    text = re.sub(r"<.*?>", " ", text)
    # Xóa số điện thoại giả dạng liên hệ spam (ví dụ 09xxxx, 0987654321)
    text = re.sub(r"\b0\d{9,10}\b", " sdt_lien_he ", text)
    # Chuẩn hóa teencode
    text = replace_teencode(text)
    
    if remove_emoji:
        try:
            import emoji
            text = emoji.replace_emoji(text, replace=" ")
        except ImportError:
            # Fallback nếu chưa cài thư viện emoji
            text = re.sub(r"[^\w\s,.?!:;\-_]", " ", text)
            
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    return text

def segment_vietnamese(text: str) -> str:
    """
    Tách từ tiếng Việt bằng Underthesea.
    Ví dụ: 'học sâu và thị giác máy tính' -> 'học_sâu và thị_giác máy_tính'
    Đây là bước tiền đề tối quan trọng trước khi đưa vào PhoBERT.
    """
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except ImportError:
        # Dự phòng trường hợp underthesea chưa cài đặt
        return text

def preprocess_pipeline(
    input_csv: str = "data/raw/data.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42
):
    """
    Quy trình tiền xử lý toàn diện:
    1. Đọc và lọc sạch null/trùng lặp
    2. Chuẩn hóa và làm sạch văn bản
    3. Tách từ
    4. Chuẩn hóa đặc trưng metadata (log transform)
    5. Chia Stratified Train / Val / Test
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Không tìm thấy file: {input_csv}. Hãy chuẩn bị dữ liệu raw trước!")

    print(f"[1/5] Đang đọc file: {input_csv} ...")
    df = pd.read_csv(input_csv)
    
    initial_len = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"])
    print(f" -> Loại bỏ trùng lặp và null: {initial_len} -> {len(df)} mẫu")

    print("[2/5] Đang làm sạch và chuẩn hóa teencode...")
    df["text_clean"] = df["text"].apply(clean_social_text)

    print("[3/5] Đang tách từ tiếng Việt (Word Segmentation)...")
    df["text_segmented"] = df["text_clean"].apply(segment_vietnamese)

    # Xử lý metadata số học (nếu có các cột likes, shares, comments)
    meta_cols = ["likes", "shares", "comments", "is_verified"]
    existing_meta = [c for c in meta_cols if c in df.columns]
    
    if existing_meta:
        print(f"[4/5] Chuẩn hóa metadata số ({existing_meta}) bằng Log1p scaling...")
        for col in ["likes", "shares", "comments"]:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].fillna(0).astype(float))
        if "is_verified" in df.columns:
            df["is_verified"] = df["is_verified"].fillna(0).astype(float)

    # Chia train, val, test bảo toàn tỉ lệ nhãn (Stratified)
    print(f"[5/5] Phân chia dữ liệu: Train ({1-test_size-val_size:.0%}) / Val ({val_size:.0%}) / Test ({test_size:.0%})...")
    
    # Chia test trước
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_seed
    )
    # Chia tiếp val từ phần còn lại
    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=adjusted_val_size, stratify=train_val_df["label"], random_state=random_seed
    )

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print(f"-> Hoàn tất! Đã lưu:")
    print(f"   + Train: {train_path} ({len(train_df)} mẫu)")
    print(f"   + Val:   {val_path} ({len(val_df)} mẫu)")
    print(f"   + Test:  {test_path} ({len(test_df)} mẫu)")

if __name__ == "__main__":
    preprocess_pipeline()
